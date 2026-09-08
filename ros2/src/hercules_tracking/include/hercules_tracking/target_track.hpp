#pragma once

#include <map>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "hercules_tracking/target_measurement.hpp"

namespace hercules_tracking {

struct TrackConfig {
  double window_seconds{5.0};
  double process_noise_spectral_density{1.0};
  double measurement_std{0.5};
  double rho{1.0};
  int max_iterations{20};
  double tolerance{1e-3};
};

struct TrackMessage {
  std::string target_id;
  Eigen::VectorXd trajectory;
  std::vector<double> times;
  State state{State::Zero()};
  StateMatrix state_covariance{StateMatrix::Identity() * 25.0};
  bool active{false};
  std::optional<double> last_direct_observation;
};

struct TargetEstimate {
  std::string target_id;
  Position position{Position::Zero()};
  Position velocity{Position::Zero()};
  PositionMatrix covariance{PositionMatrix::Identity()};
  StateMatrix state_covariance{StateMatrix::Identity()};
  double timestamp{0.0};
  bool active{false};
  int admm_iterations{0};
  double consensus_residual{0.0};
  std::optional<Position> measurement_residual;
  std::optional<double> last_direct_observation;
};

struct HandoffMessage {
  std::string target_id;
  std::string source_id;
  std::string receiver_id;
  StateMatrix information_matrix{StateMatrix::Zero()};
  State information_vector{State::Zero()};
  double timestamp{0.0};
  std::optional<StateMatrix> state_covariance;
};

std::pair<Eigen::MatrixXd, Eigen::VectorXd> assembleWindowInformation(
    const std::vector<double>& times,
    const std::vector<std::optional<TargetMeasurement>>& measurements,
    const State& prior_mean, const StateMatrix& prior_covariance,
    double process_noise_spectral_density = 1.0, int active_tracker_count = 1,
    const std::optional<StateMatrix>& handoff_information = std::nullopt,
    const std::optional<State>& handoff_information_vector = std::nullopt);

class TargetTrack {
 public:
  TargetTrack(std::string target_id, TrackConfig config = {});

  bool begin(double timestamp, const std::optional<TargetMeasurement>& measurement,
             int active_count = 1);
  double consensusStep(const std::vector<TrackMessage>& neighbor_messages);
  TrackMessage message() const;
  TargetEstimate finalize() const;
  TargetEstimate predicted(double timestamp) const;
  std::optional<HandoffMessage> handoffMessage(double timestamp);
  bool acceptHandoff(const HandoffMessage& message);
  void seed(const TrackMessage& message, double timestamp);

  const std::string& targetId() const { return target_id_; }
  bool active() const { return active_; }
  const std::vector<double>& times() const { return times_; }
  const std::vector<std::optional<TargetMeasurement>>& measurements() const { return measurements_; }
  const Eigen::VectorXd& trajectory() const { return x_; }
  const Eigen::VectorXd& dual() const { return dual_; }
  const Eigen::MatrixXd& information() const { return information_; }
  const Eigen::VectorXd& informationVector() const { return information_vector_; }
  const Eigen::MatrixXd& covariance() const { return covariance_; }
  double consensusResidual() const { return consensus_residual_; }
  int admmIterations() const { return admm_iterations_; }
  bool hasPendingHandoff() const { return pending_handoff_information_.has_value(); }
  std::optional<double> lastDirectObservation() const { return last_direct_observation_; }
  std::optional<double> lastConsensusSupport() const { return last_consensus_support_; }
  void setConsensusSupport(double timestamp, const StateMatrix& covariance);

 private:
  bool ensureInitialized(const std::optional<TargetMeasurement>& measurement, double timestamp);
  void appendSample(double timestamp, const std::optional<TargetMeasurement>& measurement);
  State latestState() const;
  void clearForFreshObservation();

  std::string target_id_;
  TrackConfig config_;
  std::optional<State> prior_mean_;
  StateMatrix prior_covariance_{StateMatrix::Identity() * 25.0};
  std::vector<double> times_;
  std::vector<std::optional<TargetMeasurement>> measurements_;
  Eigen::VectorXd x_;
  Eigen::VectorXd dual_;
  Eigen::MatrixXd information_;
  Eigen::VectorXd information_vector_;
  Eigen::MatrixXd covariance_{StateMatrix::Identity() * 25.0};
  std::optional<StateMatrix> consensus_covariance_;
  std::optional<double> latest_time_;
  std::optional<double> last_direct_observation_;
  bool active_{false};
  int admm_iterations_{0};
  double consensus_residual_{std::numeric_limits<double>::infinity()};
  std::vector<std::pair<std::string, double>> handoffs_;
  std::optional<StateMatrix> pending_handoff_information_;
  std::optional<State> pending_handoff_vector_;
  std::optional<double> last_handoff_time_;
  std::optional<double> last_handoff_sent_;
  std::optional<double> last_consensus_support_;
};

}  // namespace hercules_tracking
