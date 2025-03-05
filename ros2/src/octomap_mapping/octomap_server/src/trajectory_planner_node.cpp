#include <chrono>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "std_msgs/msg/header.hpp"
#include "octomap_msgs/msg/octomap.hpp"

// Include the OctoMap header from your installed library.
#include <octomap/octomap.h>

using namespace std::chrono_literals;

class TrajectoryPlanner : public rclcpp::Node {
public:
  TrajectoryPlanner() : Node("trajectory_planner")
  {
    // Declare parameters.
    this->declare_parameter("z_height", 0.25);
    this->declare_parameter("trajectory_length", 100.0);
    this->declare_parameter("square_size", 100.0);
    this->get_parameter("z_height", z_height_);
    this->get_parameter("trajectory_length", trajectory_length_);
    this->get_parameter("square_size", square_size_);

    // Define the planning area as a 2D square.
    grid_resolution_ = 0.25;  // meters per cell
    grid_origin_x_ = -square_size_ / 2.0;
    grid_origin_y_ = -square_size_ / 2.0;
    grid_width_ = static_cast<int>(square_size_ / grid_resolution_);
    grid_height_ = static_cast<int>(square_size_ / grid_resolution_);
    occupancy_grid_.assign(grid_width_ * grid_height_, 0);

    // Set up subscription to the binary octomap.
    octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_binary", 10,
      std::bind(&TrajectoryPlanner::octomap_callback, this, std::placeholders::_1));

    // Publishers for the planned path and visualization marker.
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/trajectory_marker", 10);

    // Timer to plan and publish a new trajectory every second.
    timer_ = this->create_wall_timer(1s, std::bind(&TrajectoryPlanner::timer_callback, this));

    // Initialize random generators for x and y.
    rng_ = std::mt19937(rd_());
    x_dist_ = std::uniform_real_distribution<double>(grid_origin_x_, grid_origin_x_ + square_size_);
    y_dist_ = std::uniform_real_distribution<double>(grid_origin_y_, grid_origin_y_ + square_size_);

    RCLCPP_INFO(this->get_logger(), "Trajectory Planner Node Initialized");
  }

private:
  // Parameters.
  double z_height_;
  double trajectory_length_;
  double square_size_;
  double grid_resolution_;
  double grid_origin_x_, grid_origin_y_;
  int grid_width_, grid_height_;
  std::vector<int8_t> occupancy_grid_;  // 0: free, 1: occupied

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

  // Octomap callback: decode the binary message, iterate over leaves,
  // and update the occupancy grid for leaves whose z coordinate is near z_height_.
  void octomap_callback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received Octomap message, updating occupancy grid");

    // Reset occupancy grid.
    std::fill(occupancy_grid_.begin(), occupancy_grid_.end(), 0);

    try {
      // Create an OcTree using the resolution from the message.
      octomap::OcTree tree(msg->resolution);
      std::stringstream ss;
      for (auto byte : msg->data) {
        ss << static_cast<char>(byte);
      }
      tree.readBinary(ss);

      // Iterate over all leaf nodes.
      for (octomap::OcTree::leaf_iterator it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (tree.isNodeOccupied(*it)) {
          double x = it.getX();
          double y = it.getY();
          double z = it.getZ();
          // If the leaf's z coordinate is close to our fixed z_height_.
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

  // Check whether a point (x, y, z) is free according to the occupancy grid.
  bool is_free(double x, double y, double z)
  {
    int ix = static_cast<int>((x - grid_origin_x_) / grid_resolution_);
    int iy = static_cast<int>((y - grid_origin_y_) / grid_resolution_);
    if (ix < 0 || ix >= grid_width_ || iy < 0 || iy >= grid_height_) {
      return false;
    }
    return occupancy_grid_[iy * grid_width_ + ix] == 0;
  }

  // Check if the straight-line segment between p1 and p2 is free by sampling points.
  bool check_line_free(const std::tuple<double, double, double>& p1,
                       const std::tuple<double, double, double>& p2)
  {
    double x1 = std::get<0>(p1), y1 = std::get<1>(p1);
    double x2 = std::get<0>(p2), y2 = std::get<1>(p2);
    double dx = x2 - x1, dy = y2 - y1;
    double distance = std::sqrt(dx*dx + dy*dy);
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

  // Plan a collision-free trajectory by randomly sampling free points in the planning area.
  // The trajectory stops when the cumulative path length reaches trajectory_length_.
  std::vector<std::tuple<double, double, double>> plan_trajectory()
  {
    std::vector<std::tuple<double, double, double>> trajectory;
    std::tuple<double, double, double> start_point;

    // Find a random free starting point.
    while (true) {
      double start_x = x_dist_(rng_);
      double start_y = y_dist_(rng_);
      if (is_free(start_x, start_y, z_height_)) {
        start_point = std::make_tuple(start_x, start_y, z_height_);
        break;
      }
    }
    trajectory.push_back(start_point);
    auto current_point = start_point;
    double total_length = 0.0;
    int max_attempts = 1000;

    for (int i = 0; i < max_attempts; ++i) {
      if (total_length >= trajectory_length_) break;
      double next_x = x_dist_(rng_);
      double next_y = y_dist_(rng_);
      auto next_point = std::make_tuple(next_x, next_y, z_height_);
      if (is_free(next_x, next_y, z_height_) && check_line_free(current_point, next_point)) {
        double dx = std::get<0>(next_point) - std::get<0>(current_point);
        double dy = std::get<1>(next_point) - std::get<1>(current_point);
        double seg_length = std::sqrt(dx*dx + dy*dy);
        if (total_length + seg_length > trajectory_length_) {
          double remaining = trajectory_length_ - total_length;
          double scale = remaining / seg_length;
          double trimmed_x = std::get<0>(current_point) + dx * scale;
          double trimmed_y = std::get<1>(current_point) + dy * scale;
          auto trimmed_point = std::make_tuple(trimmed_x, trimmed_y, z_height_);
          if (!check_line_free(current_point, trimmed_point))
            continue;
          trajectory.push_back(trimmed_point);
          total_length += remaining;
          break;
        }
        trajectory.push_back(next_point);
        total_length += seg_length;
        current_point = next_point;
      }
    }
    RCLCPP_INFO(this->get_logger(), "Planned trajectory length: %.2f m with %zu waypoints", total_length, trajectory.size());
    return trajectory;
  }

  // Publish the trajectory as a nav_msgs::msg::Path and as a visualization_msgs::msg::Marker.
  void publish_trajectory(const std::vector<std::tuple<double, double, double>> &trajectory)
  {
    auto now = this->now();
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = now;
    path_msg.header.frame_id = "map";
    for (const auto &pt : trajectory) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path_msg.header;
      pose.pose.position.x = std::get<0>(pt);
      pose.pose.position.y = std::get<1>(pt);
      pose.pose.position.z = std::get<2>(pt);
      pose.pose.orientation.w = 1.0;
      path_msg.poses.push_back(pose);
    }
    path_pub_->publish(path_msg);

    visualization_msgs::msg::Marker marker;
    marker.header = path_msg.header;
    marker.ns = "trajectory";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.1;  // Line width.
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    for (const auto &pt : trajectory) {
      geometry_msgs::msg::Point p;
      p.x = std::get<0>(pt);
      p.y = std::get<1>(pt);
      p.z = std::get<2>(pt);
      marker.points.push_back(p);
    }
    marker_pub_->publish(marker);
  }

  // Timer callback to replan and publish trajectory periodically.
  void timer_callback()
  {
    auto trajectory = plan_trajectory();
    publish_trajectory(trajectory);
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
