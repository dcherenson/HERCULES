#include "hercules_tracking_ros/tracking_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hercules_tracking_ros {
namespace {

template <typename Matrix, typename Array>
void arrayToMatrix(const Array& source, Matrix& destination) {
  for (Eigen::Index row = 0; row < destination.rows(); ++row) {
    for (Eigen::Index column = 0; column < destination.cols(); ++column) {
      destination(row, column) = source[static_cast<std::size_t>(row * destination.cols() + column)];
    }
  }
}

template <typename Matrix, typename Array>
void matrixToArray(const Matrix& source, Array& destination) {
  for (Eigen::Index row = 0; row < source.rows(); ++row) {
    for (Eigen::Index column = 0; column < source.cols(); ++column) {
      destination[static_cast<std::size_t>(row * source.cols() + column)] = source(row, column);
    }
  }
}

}  // namespace

double timeSeconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<double>(stamp.sec) + 1e-9 * static_cast<double>(stamp.nanosec);
}

builtin_interfaces::msg::Time timeMessage(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0) {
    throw std::invalid_argument("timestamp must be finite and nonnegative");
  }
  builtin_interfaces::msg::Time output;
  const double whole = std::floor(seconds);
  output.sec = static_cast<int32_t>(whole);
  output.nanosec = static_cast<uint32_t>(std::llround((seconds - whole) * 1e9));
  if (output.nanosec == 1000000000U) {
    ++output.sec;
    output.nanosec = 0;
  }
  return output;
}

hercules_tracking::TargetMeasurement measurementFromRos(
    const hercules_interfaces::msg::TargetMeasurement& message) {
  hercules_tracking::Position position(message.position[0], message.position[1]);
  hercules_tracking::PositionMatrix covariance;
  arrayToMatrix(message.covariance, covariance);
  return {message.target_id, position, covariance, timeSeconds(message.stamp), message.valid,
          message.source_id, message.capture_id.empty() ? std::nullopt
                                                        : std::optional<std::string>(message.capture_id),
          message.sensor_id.empty() ? std::nullopt
                                    : std::optional<std::string>(message.sensor_id),
          message.visible};
}

hercules_interfaces::msg::TrackingConsensus consensusToRos(
    const std::string& sender_id, uint64_t epoch_id, uint32_t round_id,
    const hercules_tracking::TrackMessage& value, double estimate_stamp) {
  hercules_interfaces::msg::TrackingConsensus output;
  output.sender_id = sender_id;
  output.target_id = value.target_id;
  output.epoch_id = epoch_id;
  output.round_id = round_id;
  output.estimate_stamp = timeMessage(estimate_stamp);
  output.trajectory_epoch_times = value.times;
  output.trajectory_state.resize(static_cast<std::size_t>(value.trajectory.size()));
  for (Eigen::Index index = 0; index < value.trajectory.size(); ++index) {
    output.trajectory_state[static_cast<std::size_t>(index)] = value.trajectory[index];
  }
  matrixToArray(value.state_covariance, output.state_covariance);
  output.active = value.active;
  return output;
}

hercules_tracking::TrackMessage consensusFromRos(
    const hercules_interfaces::msg::TrackingConsensus& message) {
  if (message.trajectory_state.size() % hercules_tracking::kStateDim != 0) {
    throw std::invalid_argument("trajectory_state must contain complete states");
  }
  const std::size_t count = message.trajectory_state.size() / hercules_tracking::kStateDim;
  if (!message.trajectory_epoch_times.empty() && message.trajectory_epoch_times.size() != count) {
    throw std::invalid_argument("trajectory times and states disagree");
  }
  hercules_tracking::TrackMessage output;
  output.target_id = message.target_id;
  output.trajectory.resize(static_cast<Eigen::Index>(message.trajectory_state.size()));
  for (std::size_t index = 0; index < message.trajectory_state.size(); ++index) {
    output.trajectory[static_cast<Eigen::Index>(index)] = message.trajectory_state[index];
  }
  output.times = message.trajectory_epoch_times;
  if (output.trajectory.size() >= hercules_tracking::kStateDim) {
    output.state = output.trajectory.tail(hercules_tracking::kStateDim);
  }
  arrayToMatrix(message.state_covariance, output.state_covariance);
  output.active = message.active;
  return output;
}

hercules_interfaces::msg::TargetEstimate estimateToRos(
    const hercules_tracking::TargetEstimate& value) {
  hercules_interfaces::msg::TargetEstimate output;
  output.target_id = value.target_id;
  output.stamp = timeMessage(value.timestamp);
  output.position = {value.position.x(), value.position.y()};
  output.velocity = {value.velocity.x(), value.velocity.y()};
  matrixToArray(value.state_covariance, output.state_covariance);
  output.active = value.active;
  output.consensus_iterations = static_cast<uint32_t>(std::max(0, value.admm_iterations));
  output.consensus_residual = value.consensus_residual;
  output.has_measurement_residual = value.measurement_residual.has_value();
  if (value.measurement_residual) {
    output.measurement_residual = {value.measurement_residual->x(), value.measurement_residual->y()};
  }
  return output;
}

hercules_interfaces::msg::TrackingHandoff handoffToRos(
    uint64_t epoch_id, const hercules_tracking::HandoffMessage& value) {
  hercules_interfaces::msg::TrackingHandoff output;
  output.sender_id = value.source_id;
  output.receiver_id = value.receiver_id;
  output.target_id = value.target_id;
  output.epoch_id = epoch_id;
  output.stamp = timeMessage(value.timestamp);
  matrixToArray(value.information_matrix, output.information_matrix);
  for (Eigen::Index index = 0; index < hercules_tracking::kStateDim; ++index) {
    output.information_vector[static_cast<std::size_t>(index)] = value.information_vector[index];
  }
  output.has_state_covariance = value.state_covariance.has_value();
  if (value.state_covariance) matrixToArray(*value.state_covariance, output.state_covariance);
  return output;
}

hercules_tracking::HandoffMessage handoffFromRos(
    const hercules_interfaces::msg::TrackingHandoff& message) {
  hercules_tracking::HandoffMessage output;
  output.source_id = message.sender_id;
  output.receiver_id = message.receiver_id;
  output.target_id = message.target_id;
  output.timestamp = timeSeconds(message.stamp);
  arrayToMatrix(message.information_matrix, output.information_matrix);
  for (Eigen::Index index = 0; index < hercules_tracking::kStateDim; ++index) {
    output.information_vector[index] = message.information_vector[static_cast<std::size_t>(index)];
  }
  if (message.has_state_covariance) {
    hercules_tracking::StateMatrix covariance;
    arrayToMatrix(message.state_covariance, covariance);
    output.state_covariance = covariance;
  }
  return output;
}

}  // namespace hercules_tracking_ros
