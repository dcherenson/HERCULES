#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

namespace hercules_mission_ros {

inline bool missionReady(const std::vector<std::string>& required,
                         const std::set<std::string>& valid_states,
                         const std::set<std::string>& calibrated_origins,
                         const std::map<std::string, double>& state_ages,
                         double freshness_timeout) {
  for (const auto& id : required) {
    const auto age = state_ages.find(id);
    if (!valid_states.count(id) || !calibrated_origins.count(id) ||
        age == state_ages.end() || age->second > freshness_timeout) return false;
  }
  return true;
}

inline bool actuationAllowed(bool dry_run, bool ready) {
  return !dry_run && ready;
}

}  // namespace hercules_mission_ros
