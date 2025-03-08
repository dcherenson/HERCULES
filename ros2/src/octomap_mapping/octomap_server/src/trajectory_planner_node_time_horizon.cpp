#include <chrono>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
#include <sstream>
#include <tuple>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "std_msgs/msg/header.hpp"
#include "octomap_msgs/msg/octomap.hpp"

#include <octomap/octomap.h>

using namespace std::chrono_literals;

class TrajectoryPlanner : public rclcpp::Node {
public:
  TrajectoryPlanner() : Node("trajectory_planner")
  {
    // Declare parameters for spatial planning.
    this->declare_parameter("z_height", 0.25);
    this->declare_parameter("square_size", 100.0);  // planning area side (meters)

    // Time horizon based planning parameters.
    this->declare_parameter("planning_horizon", 50.0); // total planning time (seconds)
    this->declare_parameter("dt", 1.0);                // time step (seconds)

    // Dynamic constraints.
    this->declare_parameter("max_linear_velocity", 2.0);     // m/s (max forward speed)
    this->declare_parameter("min_linear_velocity", 0.1);     // m/s (minimum to ensure forward motion)
    this->declare_parameter("max_angular_velocity", 0.7854); // rad/s (45 deg/s)

    // New parameters for the starting point.
    this->declare_parameter("start_x", 0.0);
    this->declare_parameter("start_y", 0.0);
    this->declare_parameter("start_z", 0.25);  // same as z_height

    // Retrieve parameter values.
    this->get_parameter("z_height", z_height_);
    this->get_parameter("square_size", square_size_);
    this->get_parameter("planning_horizon", planning_horizon_);
    this->get_parameter("dt", dt_);
    this->get_parameter("max_linear_velocity", max_linear_velocity_);
    this->get_parameter("min_linear_velocity", min_linear_velocity_);
    this->get_parameter("max_angular_velocity", max_angular_velocity_);
    this->get_parameter("start_x", start_x_);
    this->get_parameter("start_y", start_y_);
    this->get_parameter("start_z", start_z_);

    // Define the planning area as a 2D square.
    grid_resolution_ = 0.25;  // meters per cell
    grid_origin_x_ = -square_size_ / 2.0;
    grid_origin_y_ = -square_size_ / 2.0;
    grid_width_ = static_cast<int>(square_size_ / grid_resolution_);
    grid_height_ = static_cast<int>(square_size_ / grid_resolution_);
    occupancy_grid_.assign(grid_width_ * grid_height_, 0);

    // Subscribe to the binary octomap.
    octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_binary", 10,
      std::bind(&TrajectoryPlanner::octomap_callback, this, std::placeholders::_1));

    // Publishers for the planned path and visualization marker.
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/trajectory_marker", 10);

    // Timer to publish the trajectory (replanned every second).
    timer_ = this->create_wall_timer(1s, std::bind(&TrajectoryPlanner::timer_callback, this));

    // Initialize random generators.
    rng_ = std::mt19937(rd_());
    x_dist_ = std::uniform_real_distribution<double>(grid_origin_x_, grid_origin_x_ + square_size_);
    y_dist_ = std::uniform_real_distribution<double>(grid_origin_y_, grid_origin_y_ + square_size_);

    RCLCPP_INFO(this->get_logger(), "Trajectory Planner Node Initialized");
  }

private:
  // Parameters.
  double z_height_;
  double square_size_;
  double planning_horizon_;
  double dt_;
  double max_linear_velocity_;
  double min_linear_velocity_;
  double max_angular_velocity_;
  double grid_resolution_;
  double grid_origin_x_, grid_origin_y_;
  int grid_width_, grid_height_;
  std::vector<int8_t> occupancy_grid_;

  // Starting point parameters.
  double start_x_;
  double start_y_;
  double start_z_;

  // ROS publishers/subscribers/timer.
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Random number generator.
  std::random_device rd_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> x_dist_;
  std::uniform_real_distribution<double> y_dist_;

  // Storage for the planned trajectory.
  std::vector<std::tuple<double, double, double>> trajectory_;

  // Octomap callback: update the occupancy grid.
  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received Octomap message, updating occupancy grid");
    std::fill(occupancy_grid_.begin(), occupancy_grid_.end(), 0);

    try {
      octomap::OcTree tree(msg->resolution);
      std::stringstream ss;
      for (auto byte : msg->data) {
        ss << static_cast<char>(byte);
      }
      tree.readBinary(ss);

      for (octomap::OcTree::leaf_iterator it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (tree.isNodeOccupied(*it)) {
          double x = it.getX();
          double y = it.getY();
          double z = it.getZ();
          if (std::fabs(z - z_height_) < (grid_resolution_ / 2.0)) {
            int ix = static_cast<int>((x - grid_origin_x_) / grid_resolution_);
            int iy = static_cast<int>((y - grid_origin_y_) / grid_resolution_);
            if (ix >= 0 && ix < grid_width_ && iy >= 0 && iy < grid_height_) {
              occupancy_grid_[iy * grid_width_ + ix] = 1;
            }
          }
        }
      }
      RCLCPP_INFO(this->get_logger(), "Occupancy grid updated");
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error decoding Octomap: %s", e.what());
    }
  }

  // Check whether a point (x, y, z) is free.
  bool is_free(double x, double y, double z)
  {
    int ix = static_cast<int>((x - grid_origin_x_) / grid_resolution_);
    int iy = static_cast<int>((y - grid_origin_y_) / grid_resolution_);
    if (ix < 0 || ix >= grid_width_ || iy < 0 || iy >= grid_height_) {
      return false;
    }
    return (occupancy_grid_[iy * grid_width_ + ix] == 0);
  }

  // Check if the straight-line segment between two points is free.
  bool check_line_free(const std::tuple<double, double, double>& p1,
                         const std::tuple<double, double, double>& p2)
  {
    double x1 = std::get<0>(p1), y1 = std::get<1>(p1);
    double x2 = std::get<0>(p2), y2 = std::get<1>(p2);
    double dx = x2 - x1, dy = y2 - y1;
    double distance = std::sqrt(dx * dx + dy * dy);

    int steps = static_cast<int>(distance / (grid_resolution_ / 2.0));
    for (int i = 0; i <= steps; ++i) {
      double t = static_cast<double>(i) / steps;
      double x = x1 + t * dx;
      double y = y1 + t * dy;
      if (!is_free(x, y, z_height_)) {
        return false;
      }
    }
    return true;
  }

  // Generate a time-parameterized trajectory using a unicycle model.
  std::vector<std::tuple<double, double, double>> plan_trajectory()
  {
    std::vector<std::tuple<double, double, double>> traj;
    double current_x = start_x_;
    double current_y = start_y_;
    double current_theta = 0.0;  // initial heading (could be parameterized)
    double total_length = 0.0;

    // Starting point.
    auto start_point = std::make_tuple(current_x, current_y, start_z_);
    if (!is_free(current_x, current_y, start_z_)) {
      RCLCPP_WARN(this->get_logger(),
                  "Provided start point (%.2f, %.2f, %.2f) is not free!",
                  current_x, current_y, start_z_);
    }
    traj.push_back(start_point);

    // Compute the number of time steps.
    int num_steps = static_cast<int>(planning_horizon_ / dt_) + 1;
    int max_attempts_per_step = 100;

    // Random distributions for control inputs.
    std::uniform_real_distribution<double> linear_dist(min_linear_velocity_, max_linear_velocity_);
    std::uniform_real_distribution<double> angular_dist(-max_angular_velocity_, max_angular_velocity_);

    for (int i = 1; i < num_steps; ++i) {
      bool found = false;
      for (int attempt = 0; attempt < max_attempts_per_step; ++attempt) {
        // Sample control inputs: linear velocity (v) and angular velocity (omega).
        double v = linear_dist(rng_);
        double omega = angular_dist(rng_);

        // Use Euler integration to update the state.
        double new_theta = current_theta + omega * dt_;
        double next_x = current_x + v * std::cos(current_theta) * dt_;
        double next_y = current_y + v * std::sin(current_theta) * dt_;
        auto candidate = std::make_tuple(next_x, next_y, z_height_);

        // Check if the candidate is within the planning area.
        if (next_x < grid_origin_x_ || next_x > grid_origin_x_ + square_size_ ||
            next_y < grid_origin_y_ || next_y > grid_origin_y_ + square_size_) {
          continue;
        }

        // Check for collisions.
        if (is_free(next_x, next_y, z_height_) && check_line_free(traj.back(), candidate)) {
          traj.push_back(candidate);
          double step_distance = std::sqrt((next_x - current_x) * (next_x - current_x) +
                                           (next_y - current_y) * (next_y - current_y));
          total_length += step_distance;
          current_x = next_x;
          current_y = next_y;
          current_theta = new_theta;
          found = true;
          break;
        }
      }
      if (!found) {
        RCLCPP_WARN(this->get_logger(), "Could not find a valid state at time step %d", i);
        break;
      }
    }

    RCLCPP_INFO(this->get_logger(),
                "Planned trajectory length: %.2f m with %zu waypoints",
                total_length, traj.size());
    return traj;
  }

  // Publish the trajectory as a nav_msgs::Path and as a visualization Marker.
  void publish_trajectory(const std::vector<std::tuple<double, double, double>> &traj)
  {
    auto now = this->now();
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = now;
    path_msg.header.frame_id = "map";

    for (const auto &pt : traj) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path_msg.header;
      pose.pose.position.x = std::get<0>(pt);
      pose.pose.position.y = std::get<1>(pt);
      pose.pose.position.z = std::get<2>(pt);
      pose.pose.orientation.w = 1.0;  // no rotation for visualization
      path_msg.poses.push_back(pose);
    }
    path_pub_->publish(path_msg);

    visualization_msgs::msg::Marker marker;
    marker.header = path_msg.header;
    marker.ns = "trajectory";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    for (const auto &pt : traj) {
      geometry_msgs::msg::Point p;
      p.x = std::get<0>(pt);
      p.y = std::get<1>(pt);
      p.z = std::get<2>(pt);
      marker.points.push_back(p);
    }
    marker_pub_->publish(marker);
  }

  // Timer callback: generate (if not already generated) and publish the trajectory.
  void timer_callback()
  {
    if (trajectory_.empty()) {
      trajectory_ = plan_trajectory();
    }
    publish_trajectory(trajectory_);
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrajectoryPlanner>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
