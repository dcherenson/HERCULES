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
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "std_msgs/msg/header.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

using namespace std::chrono_literals;

class TrajectoryPlanner : public rclcpp::Node {
public:
  TrajectoryPlanner() : Node("trajectory_planner")
  {
    // Declare parameters.
    this->declare_parameter("z_height", -0.25);
    this->declare_parameter("trajectory_length", 200.0); // meters
    this->declare_parameter("square_size", 500.0);         // planning area side (meters)
    this->declare_parameter("max_step", 5.0);                // maximum allowed step length
    this->declare_parameter("max_linear_velocity", 2.0);     // maximum linear velocity
    this->declare_parameter("robot_name", "Husky1");

    // Starting point parameters.
    this->declare_parameter("start_x", 0.0);
    this->declare_parameter("start_y", 0.0);
    this->declare_parameter("start_z", -0.25); // default same as z_height

    // Unicycle-like constraints.
    this->declare_parameter("start_yaw", 0.0);            // initial heading in degrees
    this->declare_parameter("max_turn_angle_deg", 45.0);   // max turn angle between consecutive waypoints in degrees

    // New parameter for obstacle inflation (in meters).
    this->declare_parameter("inflation_radius", 5.0);

    // New parameter: file with required waypoints (each line: "X Y Z")
    this->declare_parameter("waypoints_file", "/home/sgarimella34/multi-robot-coordination/trajectory_data/checkpoints.txt");

    // Retrieve parameter values.
    this->get_parameter("z_height", z_height_);
    this->get_parameter("trajectory_length", trajectory_length_);
    this->get_parameter("square_size", square_size_);
    this->get_parameter("max_step", max_step_);
    this->get_parameter("max_linear_velocity", max_linear_velocity_);
    this->get_parameter("start_x", start_x_);
    this->get_parameter("start_y", start_y_);
    this->get_parameter("start_z", start_z_);
    this->get_parameter("start_yaw", start_yaw_deg_);
    this->get_parameter("max_turn_angle_deg", max_turn_angle_deg_);
    this->get_parameter("inflation_radius", inflation_radius_);
    this->get_parameter("robot_name", robot_name_);
    this->get_parameter("waypoints_file", waypoints_file_);

    output_file_path_ = "/home/sgarimella34/multi-robot-coordination/trajectory_data/trajectory_" + robot_name_ + ".txt";

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

    // Attempt to load provided waypoints if a file is specified.
    if (!waypoints_file_.empty()) {
      std::ifstream infile(waypoints_file_);
      if (!infile.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open waypoints file: %s", waypoints_file_.c_str());
      } else {
        std::string line;
        while (std::getline(infile, line)) {
          std::istringstream iss(line);
          double wx, wy, wz;
          if (iss >> wx >> wy >> wz) {
            provided_waypoints_.push_back(std::make_tuple(wx, wy, wz));
          }
        }
        infile.close();
        RCLCPP_INFO(this->get_logger(), "Loaded %zu provided waypoints.", provided_waypoints_.size());
      }
    }

    // Subscribe to the occupancy grid map published by your occupancy grid node.
    occupancy_grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/sliced_projected_map", 10,
      std::bind(&TrajectoryPlanner::occupancy_grid_callback, this, std::placeholders::_1));

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
  std::string robot_name_;
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

  // New: File name for required waypoints and storage for them.
  std::string waypoints_file_;
  std::vector<std::tuple<double, double, double>> provided_waypoints_;

  // ROS publishers/subscribers/timer.
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Random number generator.
  std::random_device rd_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> x_dist_;
  std::uniform_real_distribution<double> y_dist_;

  // Storage for the planned trajectory.
  // Each waypoint: (x, y, z, timestamp)
  std::vector<std::tuple<double, double, double, double>> trajectory_;

  // Callback to update the occupancy grid from an OccupancyGrid message.
  void occupancy_grid_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    // Reset our occupancy grid.
    std::fill(occupancy_grid_.begin(), occupancy_grid_.end(), 0);
    // For each cell in our planning area, determine occupancy based on the received grid.
    for (int j = 0; j < grid_height_; j++) {
      for (int i = 0; i < grid_width_; i++) {
        // Compute world coordinates for the center of the cell.
        double world_x = grid_origin_x_ + (i + 0.5) * grid_resolution_;
        double world_y = grid_origin_y_ + (j + 0.5) * grid_resolution_;
        // Transform world coordinates to the occupancy grid message's frame.
        int cell_x = static_cast<int>(std::floor((world_x - msg->info.origin.position.x) / msg->info.resolution));
        int cell_y = static_cast<int>(std::floor((world_y - msg->info.origin.position.y) / msg->info.resolution));
        int idx = j * grid_width_ + i;
        // Check if the computed cell is within the bounds of the received occupancy grid.
        if (cell_x >= 0 && cell_x < static_cast<int>(msg->info.width) &&
            cell_y >= 0 && cell_y < static_cast<int>(msg->info.height)) {
          int msg_index = cell_y * msg->info.width + cell_x;
          int8_t occ_value = msg->data[msg_index];
          // In the received grid, 0 is free; any nonzero value (e.g., 50 for inflation or 100 for occupied) is treated as an obstacle.
          occupancy_grid_[idx] = (occ_value == 0 ? 0 : 1);
        } else {
          occupancy_grid_[idx] = 0; // Assume free if outside the received grid.
        }
      }
    }
    RCLCPP_INFO(this->get_logger(), "Occupancy grid updated from occupancy grid map.");
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
        // If the cell is not free (nonzero) then it's considered occupied.
        if (occupancy_grid_[ny * grid_width_ + nx] != 0)
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

  // Check if the line between two waypoints is free.
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
          if (occupancy_grid_[ny * grid_width_ + nx] != 0)
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

  // Main function to plan the trajectory.
  // If provided waypoints exist, plan in segments (start->wpt1->wpt2->...->random extension);
  // otherwise, use the original random trajectory generation.
  std::vector<std::tuple<double, double, double, double>> plan_trajectory()
  {
    std::vector<std::tuple<double, double, double, double>> full_traj;
    // Start from the given starting point with time 0.0.
    full_traj.push_back(std::make_tuple(start_x_, start_y_, start_z_, 0.0));

    // We'll track the current state as we build the trajectory.
    double curr_x = start_x_, curr_y = start_y_, curr_time = 0.0, curr_theta = start_yaw_;

    // Helper lambda to compute time increment for a segment between two points.
    auto compute_dt = [this](double x0, double y0, double x1, double y1, double current_theta) -> double {
      double dx = x1 - x0, dy = y1 - y0;
      double distance = std::sqrt(dx*dx + dy*dy);
      double t_linear = distance / max_linear_velocity_;
      double new_heading = std::atan2(dy, dx);
      double dtheta = std::fabs(new_heading - current_theta);
      if (dtheta > M_PI) {
        dtheta = 2 * M_PI - dtheta;
      }
      double t_angular = (dtheta / max_turn_angle_rad_) * t_linear;
      return t_linear + t_angular;
    };

    // Lambda for planning a segment with a forced goal using recursive backtracking.
    // The base case: if a direct line from current state to goal is free, then return the goal.
    auto plan_segment = [&](double goal_x, double goal_y, double goal_z) -> std::vector<std::tuple<double,double,double,double>> {
      std::vector<std::tuple<double,double,double,double>> seg;
      // Start state is current state.
      seg.push_back(std::make_tuple(curr_x, curr_y, start_z_, curr_time));

      double seg_dx = goal_x - curr_x, seg_dy = goal_y - curr_y;
      double seg_distance = std::sqrt(seg_dx*seg_dx + seg_dy*seg_dy);
      // Determine expected number of steps for this segment.
      int expected_steps = std::max(2, static_cast<int>(std::ceil(seg_distance / max_step_)));
      double target_step = seg_distance / (expected_steps - 1);
      double min_step = target_step;
      double max_step_seg = std::min({1.5 * target_step, max_step_, max_linear_velocity_});
      int max_attempts_per_step = 500;

      // Define a recursive lambda.
      std::function<bool(int, double, double, double, double, std::vector<std::tuple<double,double,double,double>> &)> backtrackSegment;
      backtrackSegment = [&](int step_idx, double cur_x, double cur_y, double cur_theta, double cur_time,
                               std::vector<std::tuple<double,double,double,double>> &current_seg) -> bool {
        // If a direct connection from current point to goal is free, then append goal and finish.
        std::tuple<double,double,double,double> current_pt = std::make_tuple(cur_x, cur_y, start_z_, cur_time);
        std::tuple<double,double,double,double> goal_pt = std::make_tuple(goal_x, goal_y, goal_z, 0.0); // timestamp to be computed
        if (check_line_free_bresenham(current_pt, goal_pt, inflation_radius_)) {
          double dt = compute_dt(cur_x, cur_y, goal_x, goal_y, cur_theta);
          double new_time = cur_time + dt;
          current_seg.push_back(std::make_tuple(goal_x, goal_y, goal_z, new_time));
          return true;
        }
        // Otherwise, try generating an intermediate candidate.
        std::uniform_real_distribution<double> turn_dist(-max_turn_angle_rad_, max_turn_angle_rad_);
        std::uniform_real_distribution<double> step_dist(min_step, max_step_seg);
        for (int attempt = 0; attempt < max_attempts_per_step; ++attempt) {
          double dtheta = turn_dist(rng_);
          double new_theta = cur_theta + dtheta;
          double step = step_dist(rng_);
          double next_x = cur_x + step * std::cos(new_theta);
          double next_y = cur_y + step * std::sin(new_theta);
          // Check planning area boundaries.
          if (next_x < grid_origin_x_ || next_x > grid_origin_x_ + square_size_ ||
              next_y < grid_origin_y_ || next_y > grid_origin_y_ + square_size_)
            continue;
          // Check candidate cell free.
          if (!is_free(next_x, next_y, start_z_, inflation_radius_))
            continue;
          // Check line between current and candidate.
          std::tuple<double,double,double,double> candidate = std::make_tuple(next_x, next_y, start_z_, 0.0);
          if (!check_line_free_bresenham(current_seg.back(), candidate, inflation_radius_))
            continue;
          // Compute time increment.
          double dt = compute_dt(cur_x, cur_y, next_x, next_y, cur_theta);
          double new_time = cur_time + dt;
          candidate = std::make_tuple(next_x, next_y, start_z_, new_time);
          current_seg.push_back(candidate);
          if (backtrackSegment(step_idx + 1, next_x, next_y, new_theta, new_time, current_seg))
            return true;
          // Backtrack.
          current_seg.pop_back();
        }
        return false;
      };

      bool success = backtrackSegment(1, curr_x, curr_y, curr_theta, curr_time, seg);
      if (!success) {
        RCLCPP_ERROR(this->get_logger(), "Failed to plan segment to waypoint (%.2f, %.2f, %.2f)", goal_x, goal_y, goal_z);
      }
      return seg;
    };

    // Lambda for planning a random segment (without forced endpoint) for the remaining trajectory.
    auto plan_random_segment = [&](double remaining_length) -> std::vector<std::tuple<double,double,double,double>> {
      std::vector<std::tuple<double,double,double,double>> seg;
      seg.push_back(std::make_tuple(curr_x, curr_y, start_z_, curr_time));
      // Compute number of steps based on remaining length.
      int num_steps = std::max(2, static_cast<int>(std::ceil(remaining_length / max_step_)) + 1);
      double target_step = remaining_length / (num_steps - 1);
      double min_step_rand = target_step;
      double max_step_rand = std::min({1.5 * target_step, max_step_, max_linear_velocity_});
      int max_attempts_per_step = 500;
      
      std::function<bool(int, double, double, double, double, std::vector<std::tuple<double,double,double,double>> &)> backtrackRandom;
      backtrackRandom = [&](int idx, double cur_x, double cur_y, double cur_theta, double cur_time,
                              std::vector<std::tuple<double,double,double,double>> &current_seg) -> bool {
        if (idx == num_steps) {
          return true;
        }
        std::uniform_real_distribution<double> turn_dist(-max_turn_angle_rad_, max_turn_angle_rad_);
        std::uniform_real_distribution<double> step_dist(min_step_rand, max_step_rand);
        for (int attempt = 0; attempt < max_attempts_per_step; ++attempt) {
          double dtheta = turn_dist(rng_);
          double new_theta = cur_theta + dtheta;
          double step = step_dist(rng_);
          double next_x = cur_x + step * std::cos(new_theta);
          double next_y = cur_y + step * std::sin(new_theta);
          if (next_x < grid_origin_x_ || next_x > grid_origin_x_ + square_size_ ||
              next_y < grid_origin_y_ || next_y > grid_origin_y_ + square_size_)
            continue;
          if (!is_free(next_x, next_y, start_z_, inflation_radius_))
            continue;
          std::tuple<double,double,double,double> candidate = std::make_tuple(next_x, next_y, start_z_, 0.0);
          if (!check_line_free_bresenham(current_seg.back(), candidate, inflation_radius_))
            continue;
          double dt = compute_dt(cur_x, cur_y, next_x, next_y, cur_theta);
          double new_time = cur_time + dt;
          candidate = std::make_tuple(next_x, next_y, start_z_, new_time);
          current_seg.push_back(candidate);
          if (backtrackRandom(idx + 1, next_x, next_y, new_theta, new_time, current_seg))
            return true;
          current_seg.pop_back();
        }
        return false;
      };

      bool success = backtrackRandom(1, curr_x, curr_y, curr_theta, curr_time, seg);
      if (!success) {
        RCLCPP_ERROR(this->get_logger(), "Random segment backtracking failed.");
      }
      return seg;
    };

    double total_length = 0.0;
    // If provided waypoints exist, plan segments from current state to each.
    if (!provided_waypoints_.empty()) {
      for (const auto &pt : provided_waypoints_) {
        double goal_x = std::get<0>(pt);
        double goal_y = std::get<1>(pt);
        double goal_z = std::get<2>(pt);
        auto seg = plan_segment(goal_x, goal_y, goal_z);
        // Append seg (skip the first duplicate point).
        for (size_t i = 1; i < seg.size(); ++i) {
          full_traj.push_back(seg[i]);
        }
        // Update current state.
        curr_x = goal_x;
        curr_y = goal_y;
        curr_time = std::get<3>(seg.back());
        // Update heading based on the last segment.
        if (seg.size() >= 2) {
          double prev_x = std::get<0>(seg[seg.size()-2]);
          double prev_y = std::get<1>(seg[seg.size()-2]);
          curr_theta = std::atan2(goal_y - prev_y, goal_x - prev_x);
        }
      }
    }
    // Compute accumulated length so far.
    for (size_t i = 1; i < full_traj.size(); i++) {
      double dx = std::get<0>(full_traj[i]) - std::get<0>(full_traj[i - 1]);
      double dy = std::get<1>(full_traj[i]) - std::get<1>(full_traj[i - 1]);
      total_length += std::sqrt(dx*dx + dy*dy);
    }

    // If the total desired trajectory length is not yet reached, plan a final random segment.
    if (total_length < trajectory_length_) {
      double remaining_length = trajectory_length_ - total_length;
      auto seg = plan_random_segment(remaining_length);
      // Append (skipping duplicate start).
      for (size_t i = 1; i < seg.size(); ++i) {
        full_traj.push_back(seg[i]);
      }
    }
    // Log final length.
    double final_length = 0.0;
    for (size_t i = 1; i < full_traj.size(); i++) {
      double dx = std::get<0>(full_traj[i]) - std::get<0>(full_traj[i - 1]);
      double dy = std::get<1>(full_traj[i]) - std::get<1>(full_traj[i - 1]);
      final_length += std::sqrt(dx*dx + dy*dy);
    }
    RCLCPP_INFO(this->get_logger(),
                "Planned trajectory length: %.2f m with %zu waypoints",
                final_length, full_traj.size());
    return full_traj;
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
  // Each line is: "y x z t" where x and y are swapped to match the Unreal Engine coordinate system.
  void save_trajectory_to_file(const std::vector<std::tuple<double, double, double, double>> &traj)
  {
    std::ofstream ofs(output_file_path_);
    if (!ofs.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Unable to open file %s for writing", output_file_path_.c_str());
      return;
    }
    for (const auto &pt : traj) {
      ofs << std::get<1>(pt) << " "   // Y first
          << std::get<0>(pt) << " "   // X second
          << std::get<2>(pt) << " "   // Z remains
          << std::get<3>(pt) << "\n"; // Timestamp last
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
