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
#include <queue>
#include <limits>

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
  TrajectoryPlanner() : Node("trajectory_planner"), occupancy_grid_received_(false), rng_(rd_())
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

    // New parameter for obstacle inflation (meters).
    this->declare_parameter("inflation_radius", 5.0);

    // New parameter: file with required waypoints.
    this->declare_parameter("waypoints_file", "/home/sgarimella34/multi-robot-coordination/trajectory_data/checkpoints.txt");

    // Retrieve parameters.
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

    // Convert degrees to radians.
    start_yaw_ = start_yaw_deg_ * M_PI / 180.0;
    max_turn_angle_rad_ = max_turn_angle_deg_ * M_PI / 180.0;

    // Define the planning area.
    grid_resolution_ = 0.25;  // meters per cell
    grid_origin_x_ = -square_size_ / 2.0;
    grid_origin_y_ = -square_size_ / 2.0;
    grid_width_ = static_cast<int>(square_size_ / grid_resolution_);
    grid_height_ = static_cast<int>(square_size_ / grid_resolution_);
    occupancy_grid_.assign(grid_width_ * grid_height_, 0);

    // Attempt to load provided waypoints.
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

    // Subscribe to the occupancy grid.
    occupancy_grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/sliced_projected_map", 10,
      std::bind(&TrajectoryPlanner::occupancy_grid_callback, this, std::placeholders::_1));

    // Publishers.
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/trajectory_marker", 10);

    // Timer.
    timer_ = this->create_wall_timer(1s, std::bind(&TrajectoryPlanner::timer_callback, this));

    // Initialize random generators.
    x_dist_ = std::uniform_real_distribution<double>(grid_origin_x_, grid_origin_x_ + square_size_);
    y_dist_ = std::uniform_real_distribution<double>(grid_origin_y_, grid_origin_y_ + square_size_);

    RCLCPP_INFO(this->get_logger(), "Trajectory Planner Node Initialized");
  }

private:
  // Parameters and variables.
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

  // Starting point.
  double start_x_;
  double start_y_;
  double start_z_;
  double start_yaw_deg_;
  double start_yaw_;

  // Constraints.
  double max_turn_angle_deg_;
  double max_turn_angle_rad_;

  // Inflation.
  double inflation_radius_;

  // Waypoints.
  std::string waypoints_file_;
  std::vector<std::tuple<double, double, double>> provided_waypoints_;

  // Flag to ensure the occupancy grid callback runs only once.
  bool occupancy_grid_received_;

  // ROS interfaces.
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Random generators.
  std::random_device rd_;
  std::mt19937 rng_;
  std::uniform_real_distribution<double> x_dist_;
  std::uniform_real_distribution<double> y_dist_;

  // Trajectory storage.
  std::vector<std::tuple<double, double, double, double>> trajectory_;

  // ------------------ Smoothing Function ------------------
  // Applies iterative smoothing to a raw vector of 2D points.
  std::vector<std::pair<double,double>> smoothPath(const std::vector<std::pair<double,double>> &path,
                                                     int iterations = 50, double alpha = 0.1)
  {
    std::vector<std::pair<double,double>> smoothed = path;
    // Do not smooth endpoints.
    for (int iter = 0; iter < iterations; ++iter) {
      for (size_t i = 1; i < smoothed.size() - 1; ++i) {
        double new_x = smoothed[i].first + alpha * (smoothed[i-1].first + smoothed[i+1].first - 2.0 * smoothed[i].first);
        double new_y = smoothed[i].second + alpha * (smoothed[i-1].second + smoothed[i+1].second - 2.0 * smoothed[i].second);
        smoothed[i] = {new_x, new_y};
      }
    }
    return smoothed;
  }
  // --------------------------------------------------------

  // ------------------ Additional Refinement for Sharp Turns ------------------
  // For each intermediate point, if the turning angle (change in heading) is above a threshold,
  // pull that point toward the average of its neighbors.
  std::vector<std::pair<double,double>> refineSharpTurns(const std::vector<std::pair<double,double>> &path,
                                                          double turn_threshold)
  {
    std::vector<std::pair<double,double>> refined = path;
    // Process only intermediate points.
    for (size_t i = 1; i < refined.size() - 1; ++i) {
      double dx1 = refined[i].first - refined[i-1].first;
      double dy1 = refined[i].second - refined[i-1].second;
      double dx2 = refined[i+1].first - refined[i].first;
      double dy2 = refined[i+1].second - refined[i].second;
      double mag1 = std::sqrt(dx1*dx1 + dy1*dy1);
      double mag2 = std::sqrt(dx2*dx2 + dy2*dy2);
      if(mag1 < 1e-6 || mag2 < 1e-6) continue;
      double dot = dx1 * dx2 + dy1 * dy2;
      double angle = std::acos(std::clamp(dot/(mag1*mag2), -1.0, 1.0));
      // If the turning angle is large (i.e. the path is very "kinky")
      if(angle > turn_threshold) {
        // Replace this point with the average of its neighbors.
        refined[i].first = (refined[i-1].first + refined[i+1].first) / 2.0;
        refined[i].second = (refined[i-1].second + refined[i+1].second) / 2.0;
      }
    }
    return refined;
  }
  // --------------------------------------------------------

  // Callback: update occupancy grid.
  void occupancy_grid_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    // Only run this callback once after receiving a non-empty occupancy grid.
    if (occupancy_grid_received_) {
      RCLCPP_INFO(this->get_logger(), "Occupancy grid already received; ignoring new message.");
      return;
    }
    if (msg->data.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty occupancy grid; waiting for valid data.");
      return;
    }

    std::fill(occupancy_grid_.begin(), occupancy_grid_.end(), 0);
    for (int j = 0; j < grid_height_; j++) {
      for (int i = 0; i < grid_width_; i++) {
        double world_x = grid_origin_x_ + (i + 0.5) * grid_resolution_;
        double world_y = grid_origin_y_ + (j + 0.5) * grid_resolution_;
        int cell_x = static_cast<int>(std::floor((world_x - msg->info.origin.position.x) / msg->info.resolution));
        int cell_y = static_cast<int>(std::floor((world_y - msg->info.origin.position.y) / msg->info.resolution));
        int idx = j * grid_width_ + i;
        if (cell_x >= 0 && cell_x < static_cast<int>(msg->info.width) &&
            cell_y >= 0 && cell_y < static_cast<int>(msg->info.height)) {
          int msg_index = cell_y * msg->info.width + cell_x;
          int8_t occ_value = msg->data[msg_index];
          occupancy_grid_[idx] = (occ_value == 0 ? 0 : 1);
        } else {
          occupancy_grid_[idx] = 0;
        }
      }
    }
    occupancy_grid_received_ = true;
    RCLCPP_INFO(this->get_logger(), "Occupancy grid updated from occupancy grid map.");
  }

  // Bresenham algorithm (unchanged).
  std::vector<std::pair<int,int>> bresenham(int x0, int y0, int x1, int y1) {
    std::vector<std::pair<int,int>> cells;
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
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

  bool check_line_free_bresenham(const std::tuple<double,double,double,double>& p1,
                                 const std::tuple<double,double,double,double>& p2)
  {
    return check_line_free_bresenham(p1, p2, inflation_radius_);
  }

  // ---------------------- A* Algorithm Implementation ----------------------
  // Searches for a path (grid cells) from (start_x, start_y) to (goal_x, goal_y)
  std::vector<std::pair<int,int>> a_star(int start_x, int start_y, int goal_x, int goal_y)
  {
      struct PQItem {
          double f;
          int x;
          int y;
      };
      struct cmp {
          bool operator()(const PQItem &a, const PQItem &b) {
              return a.f > b.f;
          }
      };
      std::priority_queue<PQItem, std::vector<PQItem>, cmp> open;
      
      auto index = [this](int x, int y) { return y * grid_width_ + x; };
      const double INF = 1e9;
      std::vector<bool> closed(grid_width_ * grid_height_, false);
      std::vector<double> g_cost(grid_width_ * grid_height_, INF);
      std::vector<int> parent_x(grid_width_ * grid_height_, -1);
      std::vector<int> parent_y(grid_width_ * grid_height_, -1);
      
      auto heuristic = [goal_x, goal_y](int x, int y) -> double {
          return std::sqrt((x - goal_x) * (x - goal_x) + (y - goal_y) * (y - goal_y));
      };
      
      int start_idx = index(start_x, start_y);
      g_cost[start_idx] = 0.0;
      open.push({heuristic(start_x, start_y), start_x, start_y});
      
      std::vector<std::pair<int,int>> directions = {
          {1, 0}, {-1, 0}, {0, 1}, {0, -1},
          {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
      };
      
      bool found = false;
      while (!open.empty()) {
          auto current = open.top();
          open.pop();
          int cx = current.x, cy = current.y;
          int c_idx = index(cx, cy);
          if (closed[c_idx])
              continue;
          closed[c_idx] = true;
          if (cx == goal_x && cy == goal_y) {
              found = true;
              break;
          }
          for (auto d : directions) {
              int nx = cx + d.first, ny = cy + d.second;
              if (nx < 0 || nx >= grid_width_ || ny < 0 || ny >= grid_height_)
                  continue;
              int n_idx = index(nx, ny);
              if (occupancy_grid_[n_idx] != 0)
                  continue;
              if (closed[n_idx])
                  continue;
              double step_cost = (d.first != 0 && d.second != 0) ? std::sqrt(2.0) : 1.0;
              double tentative_g = g_cost[c_idx] + step_cost;
              if (tentative_g < g_cost[n_idx]) {
                  g_cost[n_idx] = tentative_g;
                  parent_x[n_idx] = cx;
                  parent_y[n_idx] = cy;
                  double f = tentative_g + heuristic(nx, ny);
                  open.push({f, nx, ny});
              }
          }
      }
      
      std::vector<std::pair<int,int>> path;
      if (!found) {
          return path;
      }
      int cx = goal_x, cy = goal_y;
      while (!(cx == start_x && cy == start_y)) {
          path.push_back({cx, cy});
          int idx_val = index(cx, cy);
          int px = parent_x[idx_val];
          int py = parent_y[idx_val];
          cx = px;
          cy = py;
      }
      path.push_back({start_x, start_y});
      std::reverse(path.begin(), path.end());
      return path;
  }
  // ------------------------------------------------------------------------

  // ---------------------- Trajectory Planning ----------------------
  std::vector<std::tuple<double, double, double, double>> plan_trajectory()
  {
      std::vector<std::tuple<double, double, double, double>> full_traj;
      // Start from the initial state.
      full_traj.push_back(std::make_tuple(start_x_, start_y_, start_z_, 0.0));

      double curr_x = start_x_, curr_y = start_y_, curr_time = 0.0, curr_theta = start_yaw_;

      // Helper: compute time increment for a segment.
      auto compute_dt = [this](double x0, double y0, double x1, double y1, double current_theta) -> double {
          double dx = x1 - x0, dy = y1 - y0;
          double distance = std::sqrt(dx * dx + dy * dy);
          double t_linear = distance / max_linear_velocity_;
          double new_heading = std::atan2(dy, dx);
          double dtheta = std::fabs(new_heading - current_theta);
          if (dtheta > M_PI) {
              dtheta = 2 * M_PI - dtheta;
          }
          double t_angular = (dtheta / max_turn_angle_rad_) * t_linear;
          return t_linear + t_angular;
      };

      // A* based segment planning from the current state to a forced waypoint.
      // This version smooths the raw A* path and then refines sharp turns.
      auto plan_segment = [&](double goal_x, double goal_y, double goal_z)
          -> std::vector<std::tuple<double,double,double,double>> {
          std::vector<std::tuple<double,double,double,double>> seg;
          seg.push_back(std::make_tuple(curr_x, curr_y, start_z_, curr_time));

          int start_cell_x = static_cast<int>(std::floor((curr_x - grid_origin_x_) / grid_resolution_));
          int start_cell_y = static_cast<int>(std::floor((curr_y - grid_origin_y_) / grid_resolution_));
          int goal_cell_x  = static_cast<int>(std::floor((goal_x - grid_origin_x_) / grid_resolution_));
          int goal_cell_y  = static_cast<int>(std::floor((goal_y - grid_origin_y_) / grid_resolution_));

          // Validate cells.
          if (start_cell_x < 0 || start_cell_x >= grid_width_ ||
              start_cell_y < 0 || start_cell_y >= grid_height_) {
              RCLCPP_ERROR(this->get_logger(), "Start cell out of bounds in A* planning.");
              return seg;
          }
          if (goal_cell_x < 0 || goal_cell_x >= grid_width_ ||
              goal_cell_y < 0 || goal_cell_y >= grid_height_) {
              RCLCPP_ERROR(this->get_logger(), "Goal cell out of bounds in A* planning.");
              return seg;
          }
          if (occupancy_grid_[start_cell_y * grid_width_ + start_cell_x] != 0) {
              RCLCPP_ERROR(this->get_logger(), "Start cell is occupied in A* planning.");
              return seg;
          }
          if (occupancy_grid_[goal_cell_y * grid_width_ + goal_cell_x] != 0) {
              RCLCPP_ERROR(this->get_logger(), "Goal cell is occupied in A* planning.");
              return seg;
          }

          auto path_cells = a_star(start_cell_x, start_cell_y, goal_cell_x, goal_cell_y);
          if (path_cells.empty()) {
              RCLCPP_ERROR(this->get_logger(), "A* failed to find a path to waypoint (%.2f, %.2f, %.2f)", goal_x, goal_y, goal_z);
              return seg;
          }

          // Convert grid cells to world coordinates.
          std::vector<std::pair<double,double>> raw_path;
          raw_path.push_back({curr_x, curr_y});
          for (size_t i = 1; i < path_cells.size(); ++i) {
              int cell_x = path_cells[i].first;
              int cell_y = path_cells[i].second;
              double wx = grid_origin_x_ + (cell_x + 0.5) * grid_resolution_;
              double wy = grid_origin_y_ + (cell_y + 0.5) * grid_resolution_;
              raw_path.push_back({wx, wy});
          }
          // Apply initial smoothing.
          auto smooth_path_result = smoothPath(raw_path, 50, 0.1);
          // Further refine sharp turns. (Threshold set to 1.0 rad; repeat a few times.)
          double turn_threshold = 1.0; // about 57 degrees
          auto refined_path = smooth_path_result;
          for (int iter = 0; iter < 5; ++iter) {
              refined_path = refineSharpTurns(refined_path, turn_threshold);
          }

          double prev_x = refined_path.front().first;
          double prev_y = refined_path.front().second;
          double time_acc = curr_time;
          double current_theta = curr_theta;

          // Convert the refined (smoothed) points into trajectory waypoints.
          for (size_t i = 1; i < refined_path.size(); ++i) {
              double wx = refined_path[i].first;
              double wy = refined_path[i].second;
              double dt = compute_dt(prev_x, prev_y, wx, wy, current_theta);
              time_acc += dt;
              seg.push_back(std::make_tuple(wx, wy, start_z_, time_acc));
              current_theta = std::atan2(wy - prev_y, wx - prev_x);
              prev_x = wx;
              prev_y = wy;
          }
          return seg;
      };

      // Random segment planning (unchanged).
      auto plan_random_segment = [&](double remaining_length)
          -> std::vector<std::tuple<double,double,double,double>> {
          std::vector<std::tuple<double,double,double,double>> seg;
          seg.push_back(std::make_tuple(curr_x, curr_y, start_z_, curr_time));
          int num_steps = std::max(2, static_cast<int>(std::ceil(remaining_length / max_step_)) + 1);
          double target_step = remaining_length / (num_steps - 1);
          double min_step_rand = target_step;
          double max_step_rand = std::min({1.5 * target_step, max_step_, max_linear_velocity_});
          int max_attempts_per_step = 500;
          
          std::function<bool(int, double, double, double, double,
                             std::vector<std::tuple<double,double,double,double>> &)> backtrackRandom;
          backtrackRandom = [&](int idx, double cur_x, double cur_y, double cur_theta, double cur_time,
                                  std::vector<std::tuple<double,double,double,double>> &current_seg) -> bool {
              if (idx == num_steps)
                  return true;
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
                  if (!check_line_free_bresenham(current_seg.back(), std::make_tuple(next_x, next_y, start_z_, 0.0), inflation_radius_))
                      continue;
                  double dt = compute_dt(cur_x, cur_y, next_x, next_y, cur_theta);
                  double new_time = cur_time + dt;
                  current_seg.push_back(std::make_tuple(next_x, next_y, start_z_, new_time));
                  if (backtrackRandom(idx + 1, next_x, next_y, new_theta, new_time, current_seg))
                      return true;
                  current_seg.pop_back();
              }
              return false;
          };

          bool success = backtrackRandom(1, curr_x, curr_y, curr_theta, curr_time, seg);
          if (!success)
              RCLCPP_ERROR(this->get_logger(), "Random segment backtracking failed.");
          return seg;
      };

      double total_length = 0.0;
      // If provided waypoints exist, plan segments via A*.
      if (!provided_waypoints_.empty()) {
          for (const auto &pt : provided_waypoints_) {
              double goal_x = std::get<0>(pt);
              double goal_y = std::get<1>(pt);
              double goal_z = std::get<2>(pt);
              auto seg = plan_segment(goal_x, goal_y, goal_z);
              for (size_t i = 1; i < seg.size(); ++i)
                  full_traj.push_back(seg[i]);
              curr_x = goal_x;
              curr_y = goal_y;
              curr_time = std::get<3>(seg.back());
              if (seg.size() >= 2) {
                  double prev_x = std::get<0>(seg[seg.size()-2]);
                  double prev_y = std::get<1>(seg[seg.size()-2]);
                  curr_theta = std::atan2(goal_y - prev_y, goal_x - prev_x);
              }
          }
      }
      // Compute accumulated trajectory length.
      for (size_t i = 1; i < full_traj.size(); i++) {
          double dx = std::get<0>(full_traj[i]) - std::get<0>(full_traj[i - 1]);
          double dy = std::get<1>(full_traj[i]) - std::get<1>(full_traj[i - 1]);
          total_length += std::sqrt(dx * dx + dy * dy);
      }
      // If more trajectory is needed, add a random segment.
      if (total_length < trajectory_length_) {
          double remaining_length = trajectory_length_ - total_length;
          auto seg = plan_random_segment(remaining_length);
          for (size_t i = 1; i < seg.size(); ++i)
              full_traj.push_back(seg[i]);
      }
      double final_length = 0.0;
      for (size_t i = 1; i < full_traj.size(); i++) {
          double dx = std::get<0>(full_traj[i]) - std::get<0>(full_traj[i - 1]);
          double dy = std::get<1>(full_traj[i]) - std::get<1>(full_traj[i - 1]);
          final_length += std::sqrt(dx * dx + dy * dy);
      }
      RCLCPP_INFO(this->get_logger(), "Planned trajectory length: %.2f m with %zu waypoints", final_length, full_traj.size());
      return full_traj;
  }
  // ------------------------------------------------------------------------

  // Publish trajectory as a Path and visualization Marker.
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

  // Save trajectory to file.
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
              << std::get<3>(pt) << "\n"; // Timestamp
      }
      ofs.close();
      RCLCPP_INFO(this->get_logger(), "Trajectory saved to %s", output_file_path_.c_str());
  }

  // Timer callback: plan (if not already done) and publish the trajectory and waypoints.
  void timer_callback()
  {
      if (!occupancy_grid_received_) {
          RCLCPP_WARN(this->get_logger(), "No occupancy grid received yet; skipping trajectory planning.");
          return;
      }
      if (trajectory_.empty()) {
          trajectory_ = plan_trajectory();
      }
      publish_trajectory(trajectory_);
      if (!trajectory_saved_) {
          save_trajectory_to_file(trajectory_);
          trajectory_saved_ = true;
      }
      // Publish waypoints as large markers if provided.
      if (!provided_waypoints_.empty()) {
          visualization_msgs::msg::Marker waypoint_marker;
          waypoint_marker.header.stamp = this->now();
          waypoint_marker.header.frame_id = "map";
          waypoint_marker.ns = "waypoints";
          waypoint_marker.id = 1;
          waypoint_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
          waypoint_marker.action = visualization_msgs::msg::Marker::ADD;
          waypoint_marker.scale.x = 2.0;
          waypoint_marker.scale.y = 2.0;
          waypoint_marker.scale.z = 2.0;
          waypoint_marker.color.a = 1.0;
          waypoint_marker.color.r = 0.0;
          waypoint_marker.color.g = 1.0;
          waypoint_marker.color.b = 0.0;
          for (const auto &pt : provided_waypoints_) {
              geometry_msgs::msg::Point p;
              p.x = std::get<0>(pt);
              p.y = std::get<1>(pt);
              p.z = std::get<2>(pt);
              waypoint_marker.points.push_back(p);
          }
          marker_pub_->publish(waypoint_marker);
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