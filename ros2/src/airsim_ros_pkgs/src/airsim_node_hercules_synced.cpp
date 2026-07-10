// airsim_node_hercules_synced.cpp
//
// A synchronized sibling to hercules_node. Instead of the async wrapper's independent
// per-stream timers against a free-running sim, this node reads pose/odom, DepthPlanar,
// segmentation and LiDAR in ONE RPC cycle and stamps every message + TF with a single
// shared timestamp. This removes the pose/image temporal skew (root cause A) and it
// requests DepthPlanar (planar Z), so the mappers need no perspective->planar conversion
// (root cause B).
//
//   sync_mode = "atomic"   : one RPC cycle per wall-timer tick, shared capture stamp.
//   sync_mode = "lockstep" : simPause + simContinueForTime(dt) per tick; everything is
//                            stamped with monotonic sim time t = step*dt. STEPS THE SIM.
//
// It re-derives the SAME topic/TF interface hercules_node exposes so hercules_nav_bridge
// works unchanged. It does NOT touch hercules_ros_wrapper / hercules_node.

#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK
#include "rpc/rpc_error.h"
STRICT_MODE_ON

#include "vehicles/car/api/CarRpcLibClient.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

using ImageCaptureBase = msr::airlib::ImageCaptureBase;
using ImageRequest = ImageCaptureBase::ImageRequest;
using ImageResponse = ImageCaptureBase::ImageResponse;
using ImageType = ImageCaptureBase::ImageType;

class HerculesSyncedNode : public rclcpp::Node
{
public:
    HerculesSyncedNode()
        : Node("hercules_node")
    {
        host_ip_ = declare_parameter<std::string>("host_ip", "localhost");
        host_port_ = declare_parameter<int>("host_port", 41452);
        vehicle_name_ = declare_parameter<std::string>("vehicle_name", "Husky1");
        camera_name_ = declare_parameter<std::string>("camera_name", "front_center");
        lidar_name_ = declare_parameter<std::string>("lidar_name", "LidarSensor1");
        sync_mode_ = declare_parameter<std::string>("sync_mode", "atomic");
        sync_dt_ = declare_parameter<double>("sync_dt", 0.05);
        publish_clock_ = declare_parameter<bool>("publish_clock", true);
        is_rgb_ = declare_parameter<bool>("is_vulkan", true); // Vulkan renderer -> rgb8
        fov_degrees_ = declare_parameter<double>("fov_degrees", 90.0);
        // Camera mount as written in settings.json (AirSim NED, metres/degrees). Only used
        // for the camera TF; the semantic mapper uses base_link_gt + a fixed offset.
        cam_x_ = declare_parameter<double>("camera_x", 0.0);
        cam_y_ = declare_parameter<double>("camera_y", -0.055);
        cam_z_ = declare_parameter<double>("camera_z", -0.35);
        cam_roll_ = declare_parameter<double>("camera_roll_deg", 0.0);
        cam_pitch_ = declare_parameter<double>("camera_pitch_deg", 0.0);
        cam_yaw_ = declare_parameter<double>("camera_yaw_deg", 0.0);

        const std::string prefix = "/hercules_node/" + vehicle_name_;
        odom_frame_id_ = vehicle_name_ + "/ground_truth/odom_local";
        lidar_frame_id_ = vehicle_name_ + "/" + lidar_name_;
        camera_body_frame_id_ = vehicle_name_ + "/" + camera_name_ + "_body";
        camera_optical_frame_id_ = vehicle_name_ + "/" + camera_name_ + "_optical";

        // QoS chosen to be compatible with the overlay's subscribers:
        //  - odom/lidar: RELIABLE (pointcloud_to_scan + odom_adapter subscribe RELIABLE)
        //  - images: BEST_EFFORT depth 1 (mappers subscribe BEST_EFFORT)
        //  - camera_info: TRANSIENT_LOCAL latch (mappers subscribe TRANSIENT_LOCAL)
        auto reliable = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
        auto sensor = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
        auto latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(prefix + "/ground_truth/odom_local", reliable);
        lidar_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(prefix + "/lidar/points/" + lidar_name_, reliable);
        depth_pub_ = create_publisher<sensor_msgs::msg::Image>(prefix + "/" + camera_name_ + "_DepthPlanar/image", sensor);
        depth_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(prefix + "/" + camera_name_ + "_DepthPlanar/camera_info", latched);
        seg_pub_ = create_publisher<sensor_msgs::msg::Image>(prefix + "/" + camera_name_ + "_Segmentation/image", sensor);
        if (publish_clock_) {
            clock_pub_ = create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::QoS(rclcpp::KeepLast(1)));
        }
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        connect();
        precompute_camera_tf();

        if (sync_mode_ == "lockstep") {
            RCLCPP_WARN(get_logger(),
                        "hercules_synced_node: LOCKSTEP mode pauses and single-steps the sim (dt=%.4f).", sync_dt_);
            client_->simPause(true);
            lockstep_thread_ = std::thread(&HerculesSyncedNode::lockstep_loop, this);
        } else {
            RCLCPP_INFO(get_logger(), "hercules_synced_node: ATOMIC mode at %.1f Hz.", 1.0 / sync_dt_);
            timer_ = create_wall_timer(
                std::chrono::duration<double>(sync_dt_),
                std::bind(&HerculesSyncedNode::atomic_cycle, this));
        }
    }

    ~HerculesSyncedNode() override
    {
        running_ = false;
        if (lockstep_thread_.joinable()) {
            lockstep_thread_.join();
        }
        try {
            if (client_ && sync_mode_ == "lockstep") {
                client_->simPause(false);
            }
        } catch (...) {
        }
    }

private:
    void connect()
    {
        client_ = std::make_unique<msr::airlib::CarRpcLibClient>(host_ip_, host_port_);
        try {
            client_->confirmConnection();
            RCLCPP_INFO(get_logger(), "hercules_synced_node connected to AirSim at %s:%d.",
                        host_ip_.c_str(), host_port_);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(get_logger(), "hercules_synced_node failed to connect: %s", e.what());
        }
    }

    // ---- one shared-timestamp cycle: pose+odom, DepthPlanar, segmentation, LiDAR ----

    void atomic_cycle()
    {
        try {
            const auto car_state = client_->getCarState(vehicle_name_);
            const rclcpp::Time stamp(static_cast<int64_t>(car_state.timestamp));
            publish_all(car_state, stamp);
        } catch (rpc::rpc_error &e) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "RPC error: %s",
                                  e.get_error().as<std::string>().c_str());
        }
    }

    void lockstep_loop()
    {
        uint64_t step = 0;
        while (running_ && rclcpp::ok()) {
            try {
                client_->simContinueForTime(sync_dt_);
                step++;
                const int64_t ns = static_cast<int64_t>(static_cast<double>(step) * sync_dt_ * 1e9);
                const rclcpp::Time stamp(ns);
                const auto car_state = client_->getCarState(vehicle_name_);
                publish_all(car_state, stamp);
            } catch (rpc::rpc_error &e) {
                RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "RPC error: %s",
                                      e.get_error().as<std::string>().c_str());
            }
        }
    }

    void publish_all(const msr::airlib::CarApiBase::CarState &car_state, const rclcpp::Time &stamp)
    {
        if (publish_clock_) {
            rosgraph_msgs::msg::Clock clk;
            clk.clock = stamp;
            clock_pub_->publish(clk);
        }
        publish_odom_and_tf(car_state, stamp);
        publish_images(stamp);
        publish_lidar(stamp);
    }

    void publish_odom_and_tf(const msr::airlib::CarApiBase::CarState &car_state, const rclcpp::Time &stamp)
    {
        const auto &k = car_state.kinematics_estimated;
        // AirSim NED -> ROS NWU (negate y, z), matching hercules_ros_wrapper.
        double px = k.pose.position.x();
        double py = -k.pose.position.y();
        double pz = -k.pose.position.z();
        double qx = k.pose.orientation.x();
        double qy = -k.pose.orientation.y();
        double qz = -k.pose.orientation.z();
        double qw = k.pose.orientation.w();

        // Pose relative to the first observed pose (start at identity), matching the
        // wrapper so the map builds around the origin instead of world coordinates.
        if (!init_received_) {
            init_px_ = px; init_py_ = py; init_pz_ = pz;
            init_q_ = tf2::Quaternion(qx, qy, qz, qw);
            init_received_ = true;
            px = py = pz = 0.0; qx = qy = qz = 0.0; qw = 1.0;
        } else {
            px -= init_px_; py -= init_py_; pz -= init_pz_;
            tf2::Quaternion q_rel = init_q_.inverse() * tf2::Quaternion(qx, qy, qz, qw);
            q_rel.normalize();
            qx = q_rel.x(); qy = q_rel.y(); qz = q_rel.z(); qw = q_rel.w();
        }

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = stamp;
        odom.header.frame_id = vehicle_name_;
        odom.child_frame_id = odom_frame_id_;
        odom.pose.pose.position.x = px;
        odom.pose.pose.position.y = py;
        odom.pose.pose.position.z = pz;
        odom.pose.pose.orientation.x = qx;
        odom.pose.pose.orientation.y = qy;
        odom.pose.pose.orientation.z = qz;
        odom.pose.pose.orientation.w = qw;
        odom.twist.twist.linear.x = k.twist.linear.x();
        odom.twist.twist.linear.y = -k.twist.linear.y();
        odom.twist.twist.linear.z = -k.twist.linear.z();
        odom.twist.twist.angular.x = k.twist.angular.x();
        odom.twist.twist.angular.y = -k.twist.angular.y();
        odom.twist.twist.angular.z = -k.twist.angular.z();
        odom_pub_->publish(odom);

        // TF, all at the shared stamp so the overlay's timestamped lookups match exactly.
        geometry_msgs::msg::TransformStamped odom_tf;
        odom_tf.header.stamp = stamp;
        odom_tf.header.frame_id = vehicle_name_;
        odom_tf.child_frame_id = odom_frame_id_;
        odom_tf.transform.translation.x = px;
        odom_tf.transform.translation.y = py;
        odom_tf.transform.translation.z = pz;
        odom_tf.transform.rotation = odom.pose.pose.orientation;
        tf_broadcaster_->sendTransform(odom_tf);

        publish_sensor_mount_tfs(stamp);
    }

    void publish_sensor_mount_tfs(const rclcpp::Time &stamp)
    {
        // LiDAR mount: query the pose once (rigid) and cache it in ROS convention.
        if (!lidar_mount_ready_) {
            try {
                const auto ld = client_->getLidarData(lidar_name_, vehicle_name_);
                lidar_tx_ = ld.pose.position.x();
                lidar_ty_ = -ld.pose.position.y();
                lidar_tz_ = -ld.pose.position.z();
                tf2::Quaternion lq(ld.pose.orientation.x(), -ld.pose.orientation.y(),
                                   -ld.pose.orientation.z(), ld.pose.orientation.w());
                lidar_qx_ = lq.x(); lidar_qy_ = lq.y(); lidar_qz_ = lq.z(); lidar_qw_ = lq.w();
                lidar_mount_ready_ = true;
            } catch (...) {
                return;
            }
        }
        std::vector<geometry_msgs::msg::TransformStamped> tfs;
        tfs.push_back(make_tf(stamp, odom_frame_id_, lidar_frame_id_,
                              lidar_tx_, lidar_ty_, lidar_tz_,
                              lidar_qx_, lidar_qy_, lidar_qz_, lidar_qw_));
        tfs.push_back(make_tf(stamp, odom_frame_id_, camera_body_frame_id_,
                              cam_body_tx_, cam_body_ty_, cam_body_tz_,
                              cam_body_qx_, cam_body_qy_, cam_body_qz_, cam_body_qw_));
        tfs.push_back(make_tf(stamp, camera_body_frame_id_, camera_optical_frame_id_,
                              0.0, 0.0, 0.0,
                              cam_opt_qx_, cam_opt_qy_, cam_opt_qz_, cam_opt_qw_));
        tf_broadcaster_->sendTransform(tfs);
    }

    static geometry_msgs::msg::TransformStamped make_tf(
        const rclcpp::Time &stamp, const std::string &parent, const std::string &child,
        double tx, double ty, double tz, double qx, double qy, double qz, double qw)
    {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = stamp;
        tf.header.frame_id = parent;
        tf.child_frame_id = child;
        tf.transform.translation.x = tx;
        tf.transform.translation.y = ty;
        tf.transform.translation.z = tz;
        tf.transform.rotation.x = qx;
        tf.transform.rotation.y = qy;
        tf.transform.rotation.z = qz;
        tf.transform.rotation.w = qw;
        return tf;
    }

    void precompute_camera_tf()
    {
        // body frame: settings pose (NED) -> ROS (negate y,z of translation + rotation).
        tf2::Quaternion q_body;
        q_body.setRPY(cam_roll_ * M_PI / 180.0, cam_pitch_ * M_PI / 180.0, cam_yaw_ * M_PI / 180.0);
        q_body.normalize();
        cam_body_tx_ = cam_x_;
        cam_body_ty_ = -cam_y_;
        cam_body_tz_ = -cam_z_;
        cam_body_qx_ = q_body.x();
        cam_body_qy_ = -q_body.y();
        cam_body_qz_ = -q_body.z();
        cam_body_qw_ = q_body.w();
        // optical frame: body_ros * (0.5,-0.5,0.5,-0.5).
        tf2::Quaternion q_body_ros(cam_body_qx_, cam_body_qy_, cam_body_qz_, cam_body_qw_);
        tf2::Quaternion q_b2o(0.5, -0.5, 0.5, -0.5);
        q_b2o.normalize();
        tf2::Quaternion q_opt = q_body_ros * q_b2o;
        q_opt.normalize();
        cam_opt_qx_ = q_opt.x(); cam_opt_qy_ = q_opt.y(); cam_opt_qz_ = q_opt.z(); cam_opt_qw_ = q_opt.w();
    }

    void publish_images(const rclcpp::Time &stamp)
    {
        std::vector<ImageRequest> requests = {
            ImageRequest(camera_name_, ImageType::DepthPlanar, true),      // pixels_as_float
            ImageRequest(camera_name_, ImageType::Segmentation, false, false),
        };
        const std::vector<ImageResponse> responses = client_->simGetImages(requests, vehicle_name_);
        for (const auto &r : responses) {
            if (r.width == 0 || r.height == 0) {
                continue;
            }
            if (r.pixels_as_float) {
                sensor_msgs::msg::Image depth;
                depth.header.stamp = stamp;
                depth.header.frame_id = camera_optical_frame_id_;
                depth.height = r.height;
                depth.width = r.width;
                depth.encoding = "32FC1";
                depth.is_bigendian = 0;
                depth.step = r.width * sizeof(float);
                depth.data.resize(r.image_data_float.size() * sizeof(float));
                std::memcpy(depth.data.data(), r.image_data_float.data(),
                            r.image_data_float.size() * sizeof(float));
                depth_pub_->publish(depth);
                publish_camera_info(stamp, r.width, r.height);
            } else {
                sensor_msgs::msg::Image seg;
                seg.header.stamp = stamp;
                seg.header.frame_id = camera_optical_frame_id_;
                seg.height = r.height;
                seg.width = r.width;
                seg.encoding = is_rgb_ ? "rgb8" : "bgr8";
                seg.is_bigendian = 0;
                seg.step = r.width * 3;
                seg.data = r.image_data_uint8;
                seg_pub_->publish(seg);
            }
        }
    }

    void publish_camera_info(const rclcpp::Time &stamp, int width, int height)
    {
        sensor_msgs::msg::CameraInfo info;
        info.header.stamp = stamp;
        info.header.frame_id = camera_optical_frame_id_;
        info.width = width;
        info.height = height;
        const double fx = (width / 2.0) / std::tan(fov_degrees_ * M_PI / 180.0 / 2.0);
        info.k = {fx, 0.0, width / 2.0, 0.0, fx, height / 2.0, 0.0, 0.0, 1.0};
        info.p = {fx, 0.0, width / 2.0, 0.0, 0.0, fx, height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        info.d = {0.0, 0.0, 0.0, 0.0, 0.0};
        info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        info.distortion_model = "plumb_bob";
        depth_info_pub_->publish(info);
    }

    void publish_lidar(const rclcpp::Time &stamp)
    {
        msr::airlib::LidarData ld;
        try {
            ld = client_->getLidarData(lidar_name_, vehicle_name_);
        } catch (...) {
            return;
        }
        if (ld.point_cloud.size() < 3) {
            return;
        }
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.stamp = stamp;
        cloud.header.frame_id = lidar_frame_id_;
        const uint32_t n = ld.point_cloud.size() / 3;
        cloud.height = 1;
        cloud.width = n;
        cloud.fields.resize(3);
        const char *names[3] = {"x", "y", "z"};
        for (int i = 0; i < 3; ++i) {
            cloud.fields[i].name = names[i];
            cloud.fields[i].offset = i * 4;
            cloud.fields[i].datatype = sensor_msgs::msg::PointField::FLOAT32;
            cloud.fields[i].count = 1;
        }
        cloud.is_bigendian = false;
        cloud.point_step = 12;
        cloud.row_step = cloud.point_step * n;
        cloud.is_dense = true;
        cloud.data.resize(static_cast<size_t>(n) * 12);
        float *out = reinterpret_cast<float *>(cloud.data.data());
        for (uint32_t i = 0; i < n; ++i) {
            // NED -> NWU: negate y and z, matching fixPointCloud in the async wrapper.
            out[i * 3 + 0] = ld.point_cloud[i * 3 + 0];
            out[i * 3 + 1] = -ld.point_cloud[i * 3 + 1];
            out[i * 3 + 2] = -ld.point_cloud[i * 3 + 2];
        }
        lidar_pub_->publish(cloud);
    }

    // params
    std::string host_ip_, vehicle_name_, camera_name_, lidar_name_, sync_mode_;
    int host_port_;
    double sync_dt_, fov_degrees_;
    bool publish_clock_, is_rgb_;
    double cam_x_, cam_y_, cam_z_, cam_roll_, cam_pitch_, cam_yaw_;

    // frame ids
    std::string odom_frame_id_, lidar_frame_id_, camera_body_frame_id_, camera_optical_frame_id_;

    // cached mounts
    bool lidar_mount_ready_ = false;
    double lidar_tx_ = 0, lidar_ty_ = 0, lidar_tz_ = 0, lidar_qx_ = 0, lidar_qy_ = 0, lidar_qz_ = 0, lidar_qw_ = 1;
    double cam_body_tx_ = 0, cam_body_ty_ = 0, cam_body_tz_ = 0;
    double cam_body_qx_ = 0, cam_body_qy_ = 0, cam_body_qz_ = 0, cam_body_qw_ = 1;
    double cam_opt_qx_ = 0, cam_opt_qy_ = 0, cam_opt_qz_ = 0, cam_opt_qw_ = 1;

    // odom relative-to-init state
    bool init_received_ = false;
    double init_px_ = 0, init_py_ = 0, init_pz_ = 0;
    tf2::Quaternion init_q_{0, 0, 0, 1};

    std::unique_ptr<msr::airlib::CarRpcLibClient> client_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr seg_pub_;
    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::thread lockstep_thread_;
    std::atomic<bool> running_{true};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HerculesSyncedNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
