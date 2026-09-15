#include <gtest/gtest.h>

#include <Eigen/Core>

#include "hercules_mission_ros/origin_calibrator.hpp"

TEST(OriginCalibrator, LocksStableTranslation) {
  hercules_mission_ros::OriginCalibrator calibrator(3, 0.02);
  const Eigen::Vector3d local(1.0, 2.0, 3.0);
  const Eigen::Vector3d origin(10.0, -4.0, 0.5);
  EXPECT_FALSE(calibrator.add(local, local + origin));
  EXPECT_FALSE(calibrator.add(local, local + origin + Eigen::Vector3d(0.005, 0, 0)));
  EXPECT_TRUE(calibrator.add(local, local + origin - Eigen::Vector3d(0.005, 0, 0)));
  EXPECT_TRUE(calibrator.origin().isApprox(origin, 1e-12));
}

TEST(OriginCalibrator, RestartsAfterDiscontinuousCandidate) {
  hercules_mission_ros::OriginCalibrator calibrator(2, 0.02);
  EXPECT_FALSE(calibrator.add(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
  EXPECT_FALSE(calibrator.add(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()));
  EXPECT_TRUE(calibrator.add(Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones()));
  EXPECT_TRUE(calibrator.origin().isApprox(Eigen::Vector3d::Ones()));
}
