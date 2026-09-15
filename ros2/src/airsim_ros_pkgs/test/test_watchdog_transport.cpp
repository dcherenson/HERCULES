#include "command_watchdog.hpp"
#include <airsim_interfaces/msg/car_controls.hpp>
#include <rclcpp/rclcpp.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <vector>

TEST(CommandWatchdogTransport, OneLatchedRosCommandExpiresWithoutAnotherPublisherMessage) {
  using Controls = airsim_interfaces::msg::CarControls;
  const auto now = [] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  };
  auto context = std::make_shared<rclcpp::Context>();
  rclcpp::InitOptions init; init.set_domain_id(144);
  context->init(0, nullptr, init);
  rclcpp::NodeOptions options; options.context(context);
  auto node = std::make_shared<rclcpp::Node>("watchdog_transport_test", options);
  rclcpp::ExecutorOptions executor_options; executor_options.context = context;
  rclcpp::executors::SingleThreadedExecutor executor(executor_options);
  executor.add_node(node);
  hercules::CommandWatchdog watchdog;
  Controls cached;
  std::vector<Controls> dispatched;
  double receipt = -1, expired = -1;
  auto input = node->create_publisher<Controls>("/watchdog_test/command", 1);
  auto output = node->create_publisher<Controls>("/watchdog_test/dispatch", 1);
  auto commands = node->create_subscription<Controls>("/watchdog_test/command", 1,
    [&](Controls::ConstSharedPtr msg) {
      cached = *msg; receipt = now(); watchdog.received(receipt); output->publish(cached);
    });
  auto sink = node->create_subscription<Controls>("/watchdog_test/dispatch", 10,
    [&](Controls::ConstSharedPtr msg) { dispatched.push_back(*msg); });
  auto timer = node->create_wall_timer(std::chrono::milliseconds(20), [&] {
    if (watchdog.stop_if_expired(now(), .5, cached)) {
      expired = now(); output->publish(cached);
    }
  });
  bool sent = false;
  const double deadline = now() + 3;
  while (now() < deadline && (expired < 0 || now() - expired < .2)) {
    executor.spin_some();
    if (!sent && input->get_subscription_count() && output->get_subscription_count()) {
      Controls command; command.throttle = .15f; input->publish(command); sent = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  context->shutdown("test complete");
  ASSERT_TRUE(sent);
  ASSERT_EQ(dispatched.size(), 2u);
  EXPECT_FLOAT_EQ(dispatched.front().throttle, .15f);
  EXPECT_FLOAT_EQ(dispatched.back().throttle, 0);
  EXPECT_FLOAT_EQ(dispatched.back().brake, 1);
  EXPECT_GE(expired - receipt, .5);
  EXPECT_LT(expired - receipt, 1.0);
}
