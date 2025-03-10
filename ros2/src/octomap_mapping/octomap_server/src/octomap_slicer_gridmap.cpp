#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "octomap_msgs/msg/octomap.hpp"
#include "octomap_msgs/conversions.h"  // for fullMsgToMap
#include "octomap/octomap.h"
#include "nav_msgs/msg/occupancy_grid.hpp"

class OctomapToOccupancyGridNode : public rclcpp::Node
{
public:
  OctomapToOccupancyGridNode()
  : Node("octomap_to_occupancy_grid_node")
  {
    // Declare and get the altitude at which to slice the octomap
    slice_altitude_ = this->declare_parameter("slice_altitude", 0.0);

    // Subscribe to the octomap topic (adjust topic name as needed)
    subscription_ = this->create_subscription<octomap_msgs::msg::Octomap>(
      "octomap_binary", 10,
      std::bind(&OctomapToOccupancyGridNode::octomapCallback, this, std::placeholders::_1));

    // Publisher for the occupancy grid (projected map)
    publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("sliced_projected_map", 10);
  }

private:
  void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg)
  {
    // Convert the received Octomap message to an octomap::OcTree pointer.
    // fullMsgToMap returns an AbstractOcTree pointer that we dynamic_cast to an OcTree.
    octomap::OcTree* tree = dynamic_cast<octomap::OcTree*>(octomap_msgs::binaryMsgToMap(*msg));

    if (!tree) {
      RCLCPP_ERROR(this->get_logger(), "Failed to convert Octomap message to OcTree");
      return;
    }

    double resolution = tree->getResolution();
    // Set tolerance for matching the specified altitude (here half the resolution)
    double tol = resolution / 2.0;

    // Variables to compute the bounding box (min and max x,y) for cells in the slice.
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();

    // We'll collect the (x, y) coordinates and occupancy state for leaves at the specified altitude.
    std::vector<std::tuple<double, double, bool>> cellData;

    // Iterate through all leaves of the octree.
    for (octomap::OcTree::leaf_iterator it = tree->begin_leafs(), end = tree->end_leafs(); it != end; ++it) {
      // Check if the center z coordinate is within tolerance of the desired slice altitude.
      if (std::fabs(it.getZ() - slice_altitude_) <= tol) {
        double x = it.getX();
        double y = it.getY();
        // Use the octomap helper to check occupancy (by default, nodes with > 0.5 probability are considered occupied)
        bool occupied = tree->isNodeOccupied(*it);

        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);

        cellData.push_back(std::make_tuple(x, y, occupied));
      }
    }

    if (cellData.empty()) {
      RCLCPP_WARN(this->get_logger(), "No octree leaves found at the specified altitude: %f", slice_altitude_);
      delete tree;
      return;
    }

    // Compute grid dimensions. The grid resolution is taken as the octree resolution.
    int width = std::ceil((max_x - min_x) / resolution);
    int height = std::ceil((max_y - min_y) / resolution);

    // Prepare the occupancy grid message.
    nav_msgs::msg::OccupancyGrid grid;
    grid.header.stamp = this->now();
    // Use the same frame_id as the incoming octomap
    grid.header.frame_id = msg->header.frame_id;
    grid.info.resolution = resolution;
    grid.info.width = width;
    grid.info.height = height;
    // Set the origin of the grid to the minimum x,y found (z is 0 for a 2D map)
    grid.info.origin.position.x = min_x;
    grid.info.origin.position.y = min_y;
    grid.info.origin.position.z = 0.0;
    // Set the orientation to identity (no rotation)
    grid.info.origin.orientation.x = 0.0;
    grid.info.origin.orientation.y = 0.0;
    grid.info.origin.orientation.z = 0.0;
    grid.info.origin.orientation.w = 1.0;

    // Initialize the grid data with free cells (0 means free).
    grid.data.assign(width * height, 0);

    // For each cell in the slice, compute its grid cell index and mark as occupied if needed.
    for (const auto & cell : cellData) {
      double x, y;
      bool occ;
      std::tie(x, y, occ) = cell;
      if (occ) {
        // Determine the column and row in the grid.
        int col = std::floor((x - min_x) / resolution);
        int row = std::floor((y - min_y) / resolution);
        int index = row * width + col;
        if (index >= 0 && index < static_cast<int>(grid.data.size())) {
          grid.data[index] = 100;  // Occupied
        }
      }
    }

    // Publish the occupancy grid.
    publisher_->publish(grid);

    // Clean up the tree allocated by fullMsgToMap.
    delete tree;
  }

  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr publisher_;
  double slice_altitude_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OctomapToOccupancyGridNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
