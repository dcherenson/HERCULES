#include <chrono>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <functional>
#include <utility>
#include <fstream>
#include <string>

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
    // Declare parameters.
    this->declare_parameter("output_file_path", "/home/sgarimella34/multi-robot-coordination/trajectory_data/trajectory.txt");
    this->declare_parameter("z_height", -0.25);
    this->declare_parameter("trajectory_length", 1000.0); // meters
    this->declare_parameter("square_size", 500.0);         // planning area side (meters)
    // this->declare_parameter("num_waypoints", 50);          // total number of waypoints
    this->declare_parameter("max_step", 5.0);                // maximum allowed step length
    this->declare_parameter("max_linear_velocity", 2.0);     // maximum linear velocity

    // Starting point parameters.
    this->declare_parameter("start_x", 0.0);
    this->declare_parameter("start_y", 0.0);
    this->declare_parameter("start_z", -0.25); // default same as z_height

    // Unicycle-like constraints.
    this->declare_parameter("start_yaw", 0.0);            // initial heading in degrees
    this->declare_parameter("max_turn_angle_deg", 45.0);   // max turn angle between consecutive waypoints in degrees

    // New parameter for obstacle inflation (in meters).
    this->declare_parameter("inflation_radius", 5.0);

    // Retrieve parameter values.
    this->get_parameter("z_height", z_height_);
    this->get_parameter("trajectory_length", trajectory_length_);
    this->get_parameter("square_size", square_size_);
    // this->get_parameter("num_waypoints", num_waypoints_);
    this->get_parameter("max_step", max_step_);
    this->get_parameter("max_linear_velocity", max_linear_velocity_);
    this->get_parameter("start_x", start_x_);
    this->get_parameter("start_y", start_y_);
    this->get_parameter("start_z", start_z_);
    this->get_parameter("start_yaw", start_yaw_deg_);
    this->get_parameter("max_turn_angle_deg", max_turn_angle_deg_);
    this->get_parameter("inflation_radius", inflation_radius_);
    this->get_parameter("output_file_path", output_file_path_);

    // Convert degrees to radians where needed.
    start_yaw_ = start_yaw_deg_ * M_PI / 180.0;
    max_turn_angle_rad_ = max_turn_angle_deg_ * M_PI / 180.0;

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

    // Timer to publish the (single) trajectory every second.
    timer_ = this->create_wall_timer(1s, std::bind(&TrajectoryPlanner::timer_callback, this));

    // Initialize random generators (Mersenne Twister).
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
  int num_waypoints_;
  double max_step_;
  double max_linear_velocity_;
  double grid_resolution_;
  double grid_origin_x_, grid_origin_y_;
  int grid_width_, grid_height_;
  std::vector<int8_t> occupancy_grid_;

  std::string output_file_path_;
  bool trajectory_saved_ = false;

  // Starting point parameters.
  double start_x_;
  double start_y_;
  double start_z_;
  double start_yaw_deg_;
  double start_yaw_;

  // Unicycle-like constraints.
  double max_turn_angle_deg_;
  double max_turn_angle_rad_;

  // Inflation radius parameter.
  double inflation_radius_;

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
  // Now each waypoint is a 4-tuple: (x, y, z, timestamp)
  std::vector<std::tuple<double, double, double, double>> trajectory_;

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

  // Check if a candidate point is free by verifying that every cell in the circular area
  // with an effective radius (inflation radius + half grid cell) around the candidate's grid cell is free.
  bool is_free(double x, double y, double z, double infl) {
    int ix = static_cast<int>((x - grid_origin_x_) / grid_resolution_);
    int iy = static_cast<int>((y - grid_origin_y_) / grid_resolution_);
    double effective_infl = infl + grid_resolution_ / 2.0;
    int inflation_cells = static_cast<int>(std::ceil(effective_infl / grid_resolution_));
    
    for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
      for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
        double dist = std::sqrt(dx*dx + dy*dy) * grid_resolution_;
        if (dist > effective_infl)
          continue;
        int nx = ix + dx;
        int ny = iy + dy;
        if (nx < 0 || nx >= grid_width_ || ny < 0 || ny >= grid_height_)
          continue;
        if (occupancy_grid_[ny * grid_width_ + nx] == 1)
          return false;
      }
    }
    return true;
  }

  // Overloaded is_free using the default inflation radius.
  bool is_free(double x, double y, double z) {
    return is_free(x, y, z, inflation_radius_);
  }

  // Bresenham's line algorithm: compute grid cells between (x0,y0) and (x1,y1).
  std::vector<std::pair<int,int>> bresenham(int x0, int y0, int x1, int y1) {
    std::vector<std::pair<int,int>> cells;
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy; // error value e_xy
    while (true) {
      cells.push_back(std::make_pair(x0, y0));
      if (x0 == x1 && y0 == y1)
        break;
      int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
    return cells;
  }

  // Overloaded check_line_free_bresenham that works with 4-tuple waypoints.
  bool check_line_free_bresenham(const std::tuple<double,double,double,double>& p1,
                                 const std::tuple<double,double,double,double>& p2,
                                 double infl)
  {
    int x0 = static_cast<int>((std::get<0>(p1) - grid_origin_x_) / grid_resolution_);
    int y0 = static_cast<int>((std::get<1>(p1) - grid_origin_y_) / grid_resolution_);
    int x1 = static_cast<int>((std::get<0>(p2) - grid_origin_x_) / grid_resolution_);
    int y1 = static_cast<int>((std::get<1>(p2) - grid_origin_y_) / grid_resolution_);
    auto line_cells = bresenham(x0, y0, x1, y1);
    
    double effective_infl = infl + grid_resolution_ / 2.0;
    int infl_cells = static_cast<int>(std::ceil(effective_infl / grid_resolution_));
    
    for (auto cell : line_cells) {
      int cx = cell.first, cy = cell.second;
      // Check every cell in the circular neighborhood of (cx,cy).
      for (int dx = -infl_cells; dx <= infl_cells; ++dx) {
        for (int dy = -infl_cells; dy <= infl_cells; ++dy) {
          double dist = std::sqrt(dx*dx + dy*dy) * grid_resolution_;
          if (dist > effective_infl)
            continue;
          int nx = cx + dx, ny = cy + dy;
          if (nx < 0 || nx >= grid_width_ || ny < 0 || ny >= grid_height_)
            continue;
          if (occupancy_grid_[ny * grid_width_ + nx] == 1)
            return false;
        }
      }
    }
    return true;
  }

  // Overloaded version without explicit inflation parameter.
  bool check_line_free_bresenham(const std::tuple<double,double,double,double>& p1,
                                 const std::tuple<double,double,double,double>& p2)
  {
    return check_line_free_bresenham(p1, p2, inflation_radius_);
  }

  // Generate a trajectory using recursive backtracking.
  // Now each waypoint is stored as (x, y, z, timestamp).
  std::vector<std::tuple<double, double, double, double>> plan_trajectory()
  {
    std::vector<std::tuple<double, double, double, double>> traj;
    // Start from the given starting point with time 0.0.
    traj.push_back(std::make_tuple(start_x_, start_y_, start_z_, 0.0));
    
    // Compute step parameters based on desired trajectory length.
    num_waypoints_ = static_cast<int>(std::ceil(trajectory_length_ / max_step_)) + 1;
    double target_step = trajectory_length_ / static_cast<double>(num_waypoints_ - 1);
    double min_step = target_step;
    double max_step_for_dist = std::min({1.5 * target_step, max_step_, max_linear_velocity_});
    int max_attempts_per_waypoint = 500;

    // Prepare random distributions for turning and stepping.
    std::uniform_real_distribution<double> turn_dist(-max_turn_angle_rad_, max_turn_angle_rad_);
    std::uniform_real_distribution<double> step_dist(min_step, max_step_for_dist);

    // Recursive lambda for backtracking.
    // Note: now we carry current_time as an extra parameter.
    std::function<bool(int, double, double, double, double, std::vector<std::tuple<double,double,double,double>> &)> backtrackTrajectory;
    backtrackTrajectory = [&](int idx, double current_x, double current_y, double current_theta, double current_time,
                                std::vector<std::tuple<double,double,double,double>> &current_traj) -> bool {
      if (idx == num_waypoints_) {
        return true; // Full trajectory has been generated.
      }
      for (int attempt = 0; attempt < max_attempts_per_waypoint; ++attempt) {
        double dtheta = turn_dist(rng_);
        double new_theta = current_theta + dtheta;
        double step = step_dist(rng_);
        double next_x = current_x + step * std::cos(new_theta);
        double next_y = current_y + step * std::sin(new_theta);

        // Compute time increment for this segment.
        double dx = next_x - current_x;
        double dy = next_y - current_y;
        double distance = std::sqrt(dx*dx + dy*dy);
        double t_linear = distance / max_linear_velocity_;
        double new_heading = std::atan2(dy, dx);
        double dtheta_abs = std::fabs(new_heading - current_theta);
        if (dtheta_abs > M_PI) {
          dtheta_abs = 2 * M_PI - dtheta_abs;
        }
        double t_angular = (dtheta_abs / max_turn_angle_rad_) * t_linear;
        double dt = t_linear + t_angular;
        double new_time = current_time + dt;

        auto candidate = std::make_tuple(next_x, next_y, z_height_, new_time);

        // Check that the candidate is within the planning area.
        if (next_x < grid_origin_x_ || next_x > grid_origin_x_ + square_size_ ||
            next_y < grid_origin_y_ || next_y > grid_origin_y_ + square_size_)
          continue;
        // Check that the candidate is free.
        if (!is_free(next_x, next_y, z_height_, inflation_radius_))
          continue;
        // Check that the connecting line is free.
        if (!check_line_free_bresenham(current_traj.back(), candidate, inflation_radius_))
          continue;

        // Candidate is valid; add it to the trajectory.
        current_traj.push_back(candidate);
        if (backtrackTrajectory(idx + 1, next_x, next_y, new_theta, new_time, current_traj))
          return true;
        // Backtrack if the candidate did not lead to a complete trajectory.
        current_traj.pop_back();
      }
      return false; // No candidate worked for this index.
    };

    bool success = backtrackTrajectory(1, start_x_, start_y_, start_yaw_, 0.0, traj);
    if (!success) {
      RCLCPP_ERROR(this->get_logger(), "Backtracking failed to generate a complete trajectory.");
    } else {
      double total_length = 0.0;
      for (size_t i = 1; i < traj.size(); i++) {
        double dx = std::get<0>(traj[i]) - std::get<0>(traj[i - 1]);
        double dy = std::get<1>(traj[i]) - std::get<1>(traj[i - 1]);
        total_length += std::sqrt(dx*dx + dy*dy);
      }
      RCLCPP_INFO(this->get_logger(),
                  "Planned trajectory length: %.2f m with %zu waypoints",
                  total_length, traj.size());
    }
    return traj;
  }

  // Publish the trajectory as a Path message and a visualization Marker.
  // Only the (x,y,z) parts are used for visualization.
  void publish_trajectory(const std::vector<std::tuple<double, double, double, double>> &traj)
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

  // Save the trajectory to a file.
  // Now each line is: "x y z t" where t is the computed timestamp.
  void save_trajectory_to_file(const std::vector<std::tuple<double, double, double, double>> &traj)
  {
    std::ofstream ofs(output_file_path_);
    if (!ofs.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Unable to open file %s for writing", output_file_path_.c_str());
      return;
    }
    // Write each waypoint as "x y z t"
    for (const auto &pt : traj) {
      ofs << std::get<0>(pt) << " "
          << std::get<1>(pt) << " "
          << std::get<2>(pt) << " "
          << std::get<3>(pt) << "\n";
    }
    ofs.close();
    RCLCPP_INFO(this->get_logger(), "Trajectory saved to %s", output_file_path_.c_str());
  }

  // Timer callback: generate (if not already generated) and publish the trajectory.
  void timer_callback()
  {
    if (trajectory_.empty()) {
      trajectory_ = plan_trajectory();
    }
    publish_trajectory(trajectory_);

    // Save the trajectory to file only once.
    if (!trajectory_saved_) {
      save_trajectory_to_file(trajectory_);
      trajectory_saved_ = true;
    }
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
