#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "octomap_msgs/msg/octomap.hpp"
#include "octomap_msgs/conversions.h" // for binaryMsgToMap
#include "octomap/octomap.h"
#include "nav_msgs/msg/occupancy_grid.hpp"

class OctomapToOccupancyGridNode : public rclcpp::Node
{
public:
  OctomapToOccupancyGridNode()
      : Node("octomap_to_occupancy_grid_node")
  {
    // Declare and get parameters:
    slice_altitude_ = this->declare_parameter("slice_altitude", 0.0);
    inflation_radius_m_ = this->declare_parameter("inflation_radius", 2.5); // in meters
    unknown_flag_ = this->declare_parameter("unknown_flag", false);         // if true, pre-fill grid with -1 (unknown)

    // Subscribe to the octomap topic (adjust topic name as needed)
    subscription_ = this->create_subscription<octomap_msgs::msg::Octomap>(
        "octomap_binary", 10,
        std::bind(&OctomapToOccupancyGridNode::octomapCallback, this, std::placeholders::_1));

    // Publisher for the occupancy grid (projected map)
    rclcpp::QoS qos(10);
    qos.transient_local();
    publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("sliced_projected_map", qos);

    // Timer to publish the grid at 1 Hz
    timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&OctomapToOccupancyGridNode::timerCallback, this));
  }

private:
  void timerCallback()
  {
    // Only publish if a valid grid has been computed.
    if (last_grid_)
    {
      // Update the timestamp to the current time.
      last_grid_->header.stamp = this->now();
      publisher_->publish(*last_grid_);
    }
  }

  void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    // Convert the received Octomap message to an octomap::OcTree pointer.
    octomap::OcTree *tree = dynamic_cast<octomap::OcTree *>(octomap_msgs::binaryMsgToMap(*msg));
    if (!tree)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to convert Octomap message to OcTree");
      return;
    }

    double resolution = tree->getResolution();
    double tol = resolution / 2.0;

    // Retrieve the full extents of the octomap.
    double full_min_x, full_min_y, full_min_z;
    double full_max_x, full_max_y, full_max_z;
    tree->getMetricMin(full_min_x, full_min_y, full_min_z);
    tree->getMetricMax(full_max_x, full_max_y, full_max_z);

    // Compute grid dimensions using full map bounds.
    int width = std::ceil((full_max_x - full_min_x) / resolution);
    int height = std::ceil((full_max_y - full_min_y) / resolution);

    // Prepare the occupancy grid message using full extents.
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

    // Determine default cell value based on the unknown_flag.
    // If unknown_flag is true, cells are pre-filled as unknown (-1), else free (0).
    int8_t default_value = unknown_flag_ ? -1 : 0;
    grid.data.assign(width * height, default_value);

    // Gather data from leaves that are at the specified slice altitude.
    std::vector<std::tuple<double, double, bool>> cellData;
    for (octomap::OcTree::leaf_iterator it = tree->begin_leafs(), end = tree->end_leafs();
         it != end; ++it)
    {
      if (std::fabs(it.getZ() - slice_altitude_) <= tol)
      {
        double x = it.getX();
        double y = it.getY();
        bool occupied = tree->isNodeOccupied(*it);
        cellData.push_back(std::make_tuple(x, y, occupied));
      }
    }

    if (cellData.empty())
    {
      RCLCPP_WARN(this->get_logger(), "No octree leaves found at the specified altitude: %f", slice_altitude_);
      delete tree;
      return;
    }

    // Mark grid cells as occupied based on the slice data.
    for (const auto &cell : cellData)
    {
      double x, y;
      bool occ;
      std::tie(x, y, occ) = cell;
      if (occ)
      {
        // Compute grid indices relative to full map bounds.
        int col = std::floor((x - full_min_x) / resolution);
        int row = std::floor((y - full_min_y) / resolution);
        int index = row * width + col;
        if (index >= 0 && index < static_cast<int>(grid.data.size()))
        {
          grid.data[index] = 100;
        }
      }
    }

    // Apply inflation logic.
    int inflation_radius_cells = std::ceil(inflation_radius_m_ / resolution);
    std::vector<int8_t> inflated_data = grid.data;
    for (int row = 0; row < height; row++)
    {
      for (int col = 0; col < width; col++)
      {
        int index = row * width + col;
        if (grid.data[index] == 100)
        {
          for (int drow = -inflation_radius_cells; drow <= inflation_radius_cells; drow++)
          {
            for (int dcol = -inflation_radius_cells; dcol <= inflation_radius_cells; dcol++)
            {
              int nrow = row + drow;
              int ncol = col + dcol;
              if (nrow >= 0 && nrow < height && ncol >= 0 && ncol < width)
              {
                float distance = std::sqrt(drow * drow + dcol * dcol);
                if (distance <= inflation_radius_cells)
                {
                  int nindex = nrow * width + ncol;
                  if (grid.data[nindex] == default_value)
                  {
                    inflated_data[nindex] = 50;
                  }
                }
              }
            }
          }
        }
      }
    }
    grid.data = inflated_data;

    // Store the computed grid for periodic publishing.
    last_grid_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(grid);

    delete tree;
  }

  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  nav_msgs::msg::OccupancyGrid::SharedPtr last_grid_;
  double slice_altitude_;
  double inflation_radius_m_;
  bool unknown_flag_; // if true, mark unobserved/free space as unknown (-1); if false, as free (0)
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OctomapToOccupancyGridNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
