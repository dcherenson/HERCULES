#pragma once

#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <hercules_interfaces/msg/target_estimate.hpp>
#include <hercules_interfaces/msg/target_measurement.hpp>
#include <hercules_interfaces/msg/tracking_consensus.hpp>
#include <hercules_interfaces/msg/tracking_handoff.hpp>

#include "hercules_tracking/target_measurement.hpp"
#include "hercules_tracking/target_track.hpp"

namespace hercules_tracking_ros {

double timeSeconds(const builtin_interfaces::msg::Time& stamp);
builtin_interfaces::msg::Time timeMessage(double seconds);

hercules_tracking::TargetMeasurement measurementFromRos(
    const hercules_interfaces::msg::TargetMeasurement& message);
hercules_interfaces::msg::TrackingConsensus consensusToRos(
    const std::string& sender_id, uint64_t epoch_id, uint32_t round_id,
    const hercules_tracking::TrackMessage& value, double estimate_stamp);
hercules_tracking::TrackMessage consensusFromRos(
    const hercules_interfaces::msg::TrackingConsensus& message);
hercules_interfaces::msg::TargetEstimate estimateToRos(
    const hercules_tracking::TargetEstimate& value);
hercules_interfaces::msg::TrackingHandoff handoffToRos(
    uint64_t epoch_id, const hercules_tracking::HandoffMessage& value);
hercules_tracking::HandoffMessage handoffFromRos(
    const hercules_interfaces::msg::TrackingHandoff& message);

}  // namespace hercules_tracking_ros
