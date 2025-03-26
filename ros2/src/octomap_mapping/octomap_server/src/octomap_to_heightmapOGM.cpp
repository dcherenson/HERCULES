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

class OctomapToHeightmapNode : public rclcpp::Node
{
public:
    OctomapToHeightmapNode()
        : Node("octomap_to_heightmap_node")
    {
        // Declare parameters:
        // robot_vertical_clearance: above ground, if an occupied cell is found that is separated from the ground
        // and is higher than this value then it is ignored unless it is part of a continuous (tree trunk) block.
        robot_clearance_m_ = this->declare_parameter("robot_vertical_clearance", 0.5);
        // unknown_flag: if true, cells with no valid height get a default “unknown” value (-1); otherwise they are zero.
        unknown_flag_ = this->declare_parameter("unknown_flag", false);

        // Subscribe to the octomap (binary) topic.
        subscription_ = this->create_subscription<octomap_msgs::msg::Octomap>(
            "octomap_binary", 10,
            std::bind(&OctomapToHeightmapNode::octomapCallback, this, std::placeholders::_1));

        // Publisher for the heightmap occupancy grid.
        rclcpp::QoS qos(10);
        qos.transient_local();
        publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("heightmap", qos);

        // Timer to publish the grid at 1 Hz.
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&OctomapToHeightmapNode::timerCallback, this));
    }

private:
    // Publish the last computed grid.
    void timerCallback()
    {
        if (last_grid_)
        {
            last_grid_->header.stamp = this->now();
            publisher_->publish(*last_grid_);
        }
    }

    // Callback that converts the octomap into a heightmap occupancy grid.
    void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg)
    {
        // Convert the received octomap message to an octomap::OcTree.
        octomap::OcTree *tree = dynamic_cast<octomap::OcTree *>(octomap_msgs::binaryMsgToMap(*msg));
        if (!tree)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to convert Octomap message to OcTree");
            return;
        }

        double resolution = tree->getResolution();
        double tol = resolution / 2.0;

        // Retrieve full extents of the octomap.
        double full_min_x, full_min_y, full_min_z;
        double full_max_x, full_max_y, full_max_z;
        tree->getMetricMin(full_min_x, full_min_y, full_min_z);
        tree->getMetricMax(full_max_x, full_max_y, full_max_z);

        // Determine the dimensions of the occupancy grid covering the full map.
        int width = std::ceil((full_max_x - full_min_x) / resolution);
        int height = std::ceil((full_max_y - full_min_y) / resolution);

        nav_msgs::msg::OccupancyGrid grid;
        grid.header.stamp = this->now();
        grid.header.frame_id = msg->header.frame_id;
        grid.info.resolution = resolution;
        grid.info.width = width;
        grid.info.height = height;
        grid.info.origin.position.x = full_min_x;
        grid.info.origin.position.y = full_min_y;
        grid.info.origin.position.z = 0.0;
        grid.info.origin.orientation.w = 1.0;

        // Pre-fill the grid with a default value (unknown = -1 or free = 0).
        int8_t default_value = unknown_flag_ ? -1 : 0;
        grid.data.assign(width * height, default_value);

        // For each cell in the grid, cast a vertical ray (from min_z to max_z) and determine its height.
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                // Compute the world-coordinate center of the cell.
                double x = full_min_x + (col + 0.5) * resolution;
                double y = full_min_y + (row + 0.5) * resolution;

                bool ground_found = false;
                double ground_z = full_min_z;
                double final_height = full_min_z;
                double last_occ_z = 0.0;

                // Cast a vertical ray in steps of the octomap resolution.
                for (double z = full_min_z; z <= full_max_z; z += resolution)
                {
                    octomap::OcTreeNode *node = tree->search(x, y, z);
                    bool occ = (node != nullptr) && tree->isNodeOccupied(node);

                    if (occ)
                    {
                        if (!ground_found)
                        {
                            // First occupied cell: assume this is ground.
                            ground_found = true;
                            ground_z = z;
                            final_height = z;
                            last_occ_z = z;
                        }
                        else
                        {
                            // Check whether this occupied cell is continuously connected with the previous one.
                            if ((z - last_occ_z) <= (resolution * 1.1))
                            { // allow a small tolerance
                                final_height = z;
                                last_occ_z = z;
                            }
                            else
                            {
                                // A gap is detected. If the new occupancy is within the robot's clearance from ground,
                                // then it may still be considered (e.g. a narrow tree trunk). Otherwise, ignore it.
                                if ((z - ground_z) <= robot_clearance_m_)
                                {
                                    final_height = z;
                                    last_occ_z = z;
                                }
                                else
                                {
                                    // Gap too large: end the vertical scan for this cell.
                                    break;
                                }
                            }
                        }
                    }
                    else
                    {
                        // If the ground has already been found and a gap appears, you might decide to stop the ray.
                        // (Uncomment the next line if you want to stop at the first gap.)
                        // if (ground_found) break;
                    }
                } // end vertical ray

                // Write the result into the grid. If a ground was found, scale the final height to 0–100.
                int idx = row * width + col;
                if (ground_found)
                {
                    int8_t height_val = static_cast<int8_t>(std::round(((final_height - full_min_z) / (full_max_z - full_min_z)) * 100.0));
                    grid.data[idx] = height_val;
                }
                else
                {
                    grid.data[idx] = default_value;
                }
            } // end col loop
        } // end row loop

        // Save the computed grid for publishing.
        last_grid_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(grid);

        delete tree;
    }

    // Parameters.
    double robot_clearance_m_;
    bool unknown_flag_;

    // ROS 2 interfaces.
    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    // The last computed occupancy grid (heightmap).
    nav_msgs::msg::OccupancyGrid::SharedPtr last_grid_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OctomapToHeightmapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
