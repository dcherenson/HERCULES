/*
   Revised slope map calculation using a larger window and least-squares plane fitting.
   Falls back to central differences if too few valid neighbors are found.
*/

#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "octomap_msgs/msg/octomap.hpp"
#include "octomap_msgs/conversions.h" // for binaryMsgToMap
#include "octomap/octomap.h"
#include "nav_msgs/msg/occupancy_grid.hpp"

// If M_PI is not defined, define it.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class OctomapToMapsNode : public rclcpp::Node
{
public:
    OctomapToMapsNode()
        : Node("octomap_to_maps_node")
    {
        // Parameters.
        robot_clearance_m_ = this->declare_parameter("robot_vertical_clearance", 0.5);
        unknown_flag_ = this->declare_parameter("unknown_flag", false);
        // Slope threshold in m/m (gradient). For example, 0.2 means a gradient above 0.2 will be marked as obstacle.
        slope_threshold_ = this->declare_parameter("slope_threshold", 0.2);
        // Obstacle inflation radius in meters.
        obstacle_inflation_radius_m_ = this->declare_parameter("obstacle_inflation_radius", 0.5);
        // Window size for plane fitting (must be odd; default 5 means 5x5 window).
        window_size_ = this->declare_parameter("window_size", 5);

        // Subscriber to the binary octomap topic.
        subscription_ = this->create_subscription<octomap_msgs::msg::Octomap>(
            "octomap_binary", 10,
            std::bind(&OctomapToMapsNode::octomapCallback, this, std::placeholders::_1));

        // Publishers for the maps.
        rclcpp::QoS qos(10);
        qos.transient_local();
        publisher_heightmap_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("heightmap", qos);
        publisher_slopemap_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("slope_map", qos);
        publisher_obstacle_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("obstacle_map", qos);

        // Timer to publish maps periodically.
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&OctomapToMapsNode::timerCallback, this));
    }

private:
    // Timer callback: publish the latest maps.
    void timerCallback()
    {
        auto stamp = this->now();
        if (heightmap_msg_)
        {
            heightmap_msg_->header.stamp = stamp;
            publisher_heightmap_->publish(*heightmap_msg_);
        }
        if (slopemap_msg_)
        {
            slopemap_msg_->header.stamp = stamp;
            publisher_slopemap_->publish(*slopemap_msg_);
        }
        if (obstacle_msg_)
        {
            obstacle_msg_->header.stamp = stamp;
            publisher_obstacle_->publish(*obstacle_msg_);
        }
    }

    // Octomap callback: build the heightmap, slope map, and obstacle map.
    void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg)
    {
        // Convert octomap message to an OcTree.
        octomap::OcTree *tree = dynamic_cast<octomap::OcTree *>(octomap_msgs::binaryMsgToMap(*msg));
        if (!tree)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to convert Octomap message to OcTree");
            return;
        }

        double resolution = tree->getResolution();

        // Retrieve full extents of the octomap.
        double full_min_x, full_min_y, full_min_z;
        double full_max_x, full_max_y, full_max_z;
        tree->getMetricMin(full_min_x, full_min_y, full_min_z);
        tree->getMetricMax(full_max_x, full_max_y, full_max_z);

        // Determine grid dimensions.
        int grid_width = std::ceil((full_max_x - full_min_x) / resolution);
        int grid_height = std::ceil((full_max_y - full_min_y) / resolution);

        // Create the heightmap occupancy grid message for visualization (normalized to 0–100).
        nav_msgs::msg::OccupancyGrid heightmap;
        heightmap.header.frame_id = msg->header.frame_id;
        heightmap.info.resolution = resolution;
        heightmap.info.width = grid_width;
        heightmap.info.height = grid_height;
        heightmap.info.origin.position.x = full_min_x;
        heightmap.info.origin.position.y = full_min_y;
        heightmap.info.origin.position.z = 0.0;
        heightmap.info.origin.orientation.w = 1.0;
        heightmap.data.assign(grid_width * grid_height, unknown_flag_ ? -1 : 0);

        // Create a vector to hold the raw height values (in meters).
        std::vector<double> raw_heights(grid_width * grid_height, std::numeric_limits<double>::quiet_NaN());

        // For each cell, cast a vertical ray from full_min_z to full_max_z.
        for (int row = 0; row < grid_height; row++)
        {
            for (int col = 0; col < grid_width; col++)
            {
                double x = full_min_x + (col + 0.5) * resolution;
                double y = full_min_y + (row + 0.5) * resolution;
                bool ground_found = false;
                double ground_z = full_min_z;
                double final_height = full_min_z;
                double last_occ_z = 0.0;
                // Vertical ray.
                for (double z = full_min_z; z <= full_max_z; z += resolution)
                {
                    octomap::OcTreeNode *node = tree->search(x, y, z);
                    bool occ = (node != nullptr) && tree->isNodeOccupied(node);
                    if (occ)
                    {
                        if (!ground_found)
                        {
                            ground_found = true;
                            ground_z = z;
                            final_height = z;
                            last_occ_z = z;
                        }
                        else
                        {
                            if ((z - last_occ_z) <= (resolution * 1.1))
                            {
                                final_height = z;
                                last_occ_z = z;
                            }
                            else
                            {
                                if ((z - ground_z) <= robot_clearance_m_)
                                {
                                    final_height = z;
                                    last_occ_z = z;
                                }
                                else
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
                int idx = row * grid_width + col;
                if (ground_found)
                {
                    raw_heights[idx] = final_height; // store true height (in meters)
                }
                else
                {
                    raw_heights[idx] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }

        // --- Build normalized heightmap for visualization ---
        double hmin = std::numeric_limits<double>::infinity();
        double hmax = -std::numeric_limits<double>::infinity();
        for (auto h : raw_heights)
        {
            if (!std::isnan(h))
            {
                hmin = std::min(hmin, h);
                hmax = std::max(hmax, h);
            }
        }
        for (size_t i = 0; i < raw_heights.size(); i++)
        {
            if (!std::isnan(raw_heights[i]))
            {
                double norm = (raw_heights[i] - hmin) / (hmax - hmin) * 100.0;
                heightmap.data[i] = static_cast<int8_t>(std::round(norm));
            }
            else
            {
                heightmap.data[i] = unknown_flag_ ? -1 : 0;
            }
        }
        heightmap_msg_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(heightmap);

        // --- Compute the slope map using plane fitting over a larger window ---
        // Ensure the window size is odd.
        int win = window_size_;
        if (win % 2 == 0)
            win++;
        int half_win = (win - 1) / 2;

        nav_msgs::msg::OccupancyGrid slopemap;
        slopemap.header = heightmap.header;
        slopemap.info = heightmap.info;
        slopemap.data.assign(grid_width * grid_height, 0);

        // We'll keep the raw gradient (m/m) for obstacle detection.
        std::vector<double> gradient_values(grid_width * grid_height, 0.0);

        // For cells where the full window is available.
        for (int row = half_win; row < grid_height - half_win; row++)
        {
            for (int col = half_win; col < grid_width - half_win; col++)
            {
                int idx = row * grid_width + col;
                // Use the center height as reference.
                if (std::isnan(raw_heights[idx]))
                {
                    gradient_values[idx] = 0.0;
                    slopemap.data[idx] = 0;
                    continue;
                }
                double z0 = raw_heights[idx];
                // Compute the mean of neighbor positions (we can use the center of the window as reference)
                double x0 = full_min_x + (col + 0.5) * resolution;
                double y0 = full_min_y + (row + 0.5) * resolution;
                // Accumulators for least-squares fit (after subtracting center value).
                double Sxx = 0.0, Syy = 0.0, Sxy = 0.0;
                double Sxz = 0.0, Syz = 0.0;
                int count = 0;
                for (int dy = -half_win; dy <= half_win; dy++)
                {
                    for (int dx = -half_win; dx <= half_win; dx++)
                    {
                        int r = row + dy;
                        int c = col + dx;
                        int nidx = r * grid_width + c;
                        double z = raw_heights[nidx];
                        if (std::isnan(z))
                            continue;
                        // Compute relative coordinates (with respect to the center cell).
                        double x = full_min_x + (c + 0.5) * resolution;
                        double y = full_min_y + (r + 0.5) * resolution;
                        double x_rel = x - x0;
                        double y_rel = y - y0;
                        double z_rel = z - z0;
                        Sxx += x_rel * x_rel;
                        Syy += y_rel * y_rel;
                        Sxy += x_rel * y_rel;
                        Sxz += x_rel * z_rel;
                        Syz += y_rel * z_rel;
                        count++;
                    }
                }
                // Only compute if enough valid neighbors are present.
                if (count < (win * win / 2))
                {
                    // Fallback: use central differences (as in the second code).
                    double dx = (raw_heights[row * grid_width + (col + 1)] - raw_heights[row * grid_width + (col - 1)]) / (2.0 * resolution);
                    double dy = (raw_heights[(row + 1) * grid_width + col] - raw_heights[(row - 1) * grid_width + col]) / (2.0 * resolution);
                    double grad = std::sqrt(dx * dx + dy * dy);
                    gradient_values[idx] = grad;
                    double angle = std::atan(grad);
                    double normSlope = (angle / (M_PI / 2.0)) * 100.0;
                    slopemap.data[idx] = static_cast<int8_t>(std::round(normSlope));
                }
                else
                {
                    double denom = (Sxx * Syy - Sxy * Sxy);
                    double a = 0.0, b = 0.0;
                    if (std::abs(denom) > 1e-6)
                    {
                        a = (Sxz * Syy - Syz * Sxy) / denom;
                        b = (Sxx * Syz - Sxy * Sxz) / denom;
                    }
                    double grad = std::sqrt(a * a + b * b);
                    gradient_values[idx] = grad;
                    double angle = std::atan(grad);
                    double normSlope = (angle / (M_PI / 2.0)) * 100.0;
                    slopemap.data[idx] = static_cast<int8_t>(std::round(normSlope));
                }
            }
        }
        slopemap_msg_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(slopemap);

        // --- Compute the obstacle map from the gradient values ---
        // Mark cells as obstacles if the raw gradient exceeds slope_threshold_.
        nav_msgs::msg::OccupancyGrid obstaclem;
        obstaclem.header = heightmap.header;
        obstaclem.info = heightmap.info;
        int8_t def_val = unknown_flag_ ? -1 : 0;
        obstaclem.data.assign(grid_width * grid_height, def_val);
        for (int row = half_win; row < grid_height - half_win; row++)
        {
            for (int col = half_win; col < grid_width - half_win; col++)
            {
                int idx = row * grid_width + col;
                if (gradient_values[idx] > slope_threshold_)
                {
                    obstaclem.data[idx] = 100; // Mark as obstacle.
                }
            }
        }
        // Apply obstacle inflation.
        int inflation_radius_cells = std::ceil(obstacle_inflation_radius_m_ / resolution);
        std::vector<int8_t> inflated = obstaclem.data;
        for (int row = 0; row < grid_height; row++)
        {
            for (int col = 0; col < grid_width; col++)
            {
                int idx = row * grid_width + col;
                if (obstaclem.data[idx] == 100)
                {
                    for (int drow = -inflation_radius_cells; drow <= inflation_radius_cells; drow++)
                    {
                        for (int dcol = -inflation_radius_cells; dcol <= inflation_radius_cells; dcol++)
                        {
                            int nrow = row + drow;
                            int ncol = col + dcol;
                            if (nrow >= 0 && nrow < grid_height && ncol >= 0 && ncol < grid_width)
                            {
                                int nidx = nrow * grid_width + ncol;
                                if (obstaclem.data[nidx] != 100)
                                    inflated[nidx] = 50;
                            }
                        }
                    }
                }
            }
        }
        obstaclem.data = inflated;
        obstacle_msg_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(obstaclem);

        delete tree;
    }

    // Parameters.
    double robot_clearance_m_;
    bool unknown_flag_;
    double slope_threshold_;
    double obstacle_inflation_radius_m_;
    int window_size_;

    // ROS 2 interfaces.
    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_heightmap_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_slopemap_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_obstacle_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Stored map messages.
    nav_msgs::msg::OccupancyGrid::SharedPtr heightmap_msg_;
    nav_msgs::msg::OccupancyGrid::SharedPtr slopemap_msg_;
    nav_msgs::msg::OccupancyGrid::SharedPtr obstacle_msg_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OctomapToMapsNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
