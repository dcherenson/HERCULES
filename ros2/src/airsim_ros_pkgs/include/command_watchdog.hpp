#pragma once

namespace hercules {
// Wall-clock lease for latched car controls. Caller serializes access.
class CommandWatchdog {
 public:
  void received(double now) { last_ = now; active_ = true; }
  bool expired(double now, double timeout) {
    if (active_ && timeout > 0.0 && now - last_ >= timeout) {
      active_ = false;
      return true;
    }
    return false;
  }
  template <typename Controls>
  bool stop_if_expired(double now, double timeout, Controls &controls) {
    if (!expired(now, timeout)) return false;
    controls = Controls{};
    controls.brake = 1.0f;
    return true;
  }
 private:
  double last_ = 0.0;
  bool active_ = false;
};
}  // namespace hercules
