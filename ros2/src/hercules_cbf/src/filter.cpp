#include "hercules_cbf/filter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <osqp.h>

namespace hercules_cbf {
namespace {

struct CscStorage {
  std::vector<OSQPInt> p, i;
  std::vector<OSQPFloat> x;
  OSQPCscMatrix matrix{};
};

CscStorage denseToCsc(const Eigen::MatrixXd& input) {
  CscStorage storage;
  storage.p.push_back(0);
  for (Eigen::Index col = 0; col < input.cols(); ++col) {
    for (Eigen::Index row = 0; row < input.rows(); ++row) {
      if (input(row, col) == 0.0) continue;
      storage.i.push_back(static_cast<OSQPInt>(row));
      storage.x.push_back(input(row, col));
    }
    storage.p.push_back(static_cast<OSQPInt>(storage.x.size()));
  }
  storage.matrix.m = static_cast<OSQPInt>(input.rows());
  storage.matrix.n = static_cast<OSQPInt>(input.cols());
  storage.matrix.p = storage.p.data();
  storage.matrix.i = storage.i.data();
  storage.matrix.x = storage.x.data();
  storage.matrix.nzmax = static_cast<OSQPInt>(storage.x.size());
  storage.matrix.nz = -1;
  storage.matrix.owned = 0;
  return storage;
}

struct SolveOutput {
  Eigen::VectorXd control;
  bool success{false};
  std::string status{"solver_error"};
  SolverDiagnostics diagnostics;
};

SolveOutput solveOsqp(const Eigen::VectorXd& nominal, const ConstraintSet& constraints,
                      const CBFConfig& config) {
  SolveOutput output;
  const int n = static_cast<int>(nominal.size());
  const int m = static_cast<int>(constraints.rows.size());
  output.control = Eigen::VectorXd::Zero(n);
  output.diagnostics.solver = "osqp";
  Eigen::MatrixXd matrix(m, n);
  for (int row = 0; row < m; ++row) matrix.row(row) = constraints.rows[row].transpose();
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);
  auto p = denseToCsc(identity);
  auto a = denseToCsc(matrix);
  // denseToCsc returns owning storage by value; refresh the matrix views after
  // the move so the API sees the vectors' final addresses.
  p.matrix.p = p.p.data();
  p.matrix.i = p.i.data();
  p.matrix.x = p.x.data();
  a.matrix.p = a.p.data();
  a.matrix.i = a.i.data();
  a.matrix.x = a.x.data();
  std::vector<OSQPFloat> q(n), l(m), u(m, OSQP_INFTY);
  for (int j = 0; j < n; ++j) q[j] = -nominal[j];
  for (int row = 0; row < m; ++row) l[row] = constraints.rhs[row];
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.polishing = 0;
  settings.eps_abs = config.solver_eps_abs;
  settings.eps_rel = config.solver_eps_rel;
  settings.max_iter = config.solver_max_iter;
  settings.warm_starting = 0;
  OSQPSolver* solver = nullptr;
  const OSQPInt setup_status = osqp_setup(&solver, &p.matrix, q.data(), &a.matrix,
                                          l.data(), u.data(), m, n, &settings);
  if (setup_status != 0 || solver == nullptr) {
    output.status = "solver_setup_failed";
    output.diagnostics.raw_status = output.status;
    return output;
  }
  osqp_solve(solver);
  output.diagnostics.raw_status = solver->info ? solver->info->status : "solver_error";
  if (solver->info) {
    output.diagnostics.iterations = static_cast<int>(solver->info->iter);
    output.diagnostics.primal_residual = solver->info->prim_res;
    output.diagnostics.dual_residual = solver->info->dual_res;
  }
  output.status = output.diagnostics.raw_status;
  output.success = solver->info && (solver->info->status_val == OSQP_SOLVED ||
                                    solver->info->status_val == OSQP_SOLVED_INACCURATE) &&
                   solver->solution && solver->solution->x;
  if (output.success) {
    for (int j = 0; j < n; ++j) output.control[j] = solver->solution->x[j];
  }
  osqp_cleanup(solver);
  return output;
}

CBFResult failureResult(const CBFRequest& request, const CBFConfig& config,
                        const std::string& status, const std::chrono::steady_clock::time_point& start) {
  CBFResult result;
  result.safe_control = Eigen::VectorXd::Zero(controlDimension(request, config));
  result.status = status;
  result.message = "fail-safe control applied";
  result.fallback = true;
  result.solve_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  (void)config;
  return result;
}

}  // namespace

Eigen::VectorXd projectedCorrection(Eigen::VectorXd control, const ConstraintSet& constraints,
                                    const CBFConfig& config, int* rounds) {
  *rounds = 0;
  for (int round = 0; round < config.distributed_rounds; ++round) {
    const Eigen::VectorXd previous = control;
    for (std::size_t index = 0; index < constraints.rows.size(); ++index) {
      const auto& row = constraints.rows[index];
      const double violation = constraints.rhs[index] - row.dot(control);
      if (violation > 0.0) {
        control += (violation / (row.dot(row) + 1e-9)) * row;
        control = control.cwiseMax(constraints.lower).cwiseMin(constraints.upper);
      }
    }
    *rounds = round + 1;
    if ((control - previous).norm() <= config.distributed_tolerance) break;
  }
  return control;
}

double maximumRowViolation(const Eigen::VectorXd& control, const ConstraintSet& constraints) {
  double maximum = 0.0;
  for (std::size_t index = 0; index < constraints.rows.size(); ++index)
    maximum = std::max(maximum, constraints.rhs[index] - constraints.rows[index].dot(control));
  return std::max(0.0, maximum);
}

double maximumBoundViolation(const Eigen::VectorXd& control, const ConstraintSet& constraints) {
  double maximum = 0.0;
  for (Eigen::Index index = 0; index < control.size(); ++index) {
    maximum = std::max(maximum, constraints.lower[index] - control[index]);
    maximum = std::max(maximum, control[index] - constraints.upper[index]);
  }
  return std::max(0.0, maximum);
}

CBFResult filter(const CBFRequest& request, const CBFConfig& config) {
  const auto start = std::chrono::steady_clock::now();
  const int dimension = controlDimension(request, config);
  if (!request.sensor_valid) return failureResult(request, config, "invalid_or_stale_sensor", start);
  if (request.nominal_control.size() < dimension ||
      !request.nominal_control.head(dimension).array().isFinite().all())
    return failureResult(request, config, "invalid_nominal_control", start);
  const Eigen::VectorXd nominal = request.nominal_control.head(dimension);
  const ConstraintSet constraints = buildConstraints(request, config);
  SolveOutput solve;
  if (constraints.rows.empty()) {
    solve.control = nominal;
    solve.success = true;
    solve.status = "solved_no_constraints";
    solve.diagnostics.solver = "none";
    solve.diagnostics.raw_status = solve.status;
  } else {
    solve = solveOsqp(nominal, constraints, config);
  }
  CBFResult result;
  result.success = solve.success;
  result.status = solve.status;
  result.solver = solve.diagnostics;
  result.constraint_count = static_cast<int>(constraints.rows.size());
  result.robust_terms = constraints.robust_terms;
  result.minimum_barrier = constraints.barriers.empty()
      ? std::numeric_limits<double>::infinity()
      : *std::min_element(constraints.barriers.begin(), constraints.barriers.end());
  if (solve.success) {
    result.safe_control = solve.control.cwiseMax(constraints.lower).cwiseMin(constraints.upper);
  } else if (isUnicycle(request, config)) {
    result.safe_control = Eigen::VectorXd::Zero(dimension);
  } else {
    result.safe_control = (-request.ego.velocity).head(dimension);
    result.safe_control = result.safe_control.cwiseMax(
        Eigen::VectorXd::Constant(dimension, -config.uav_acceleration_limit));
    result.safe_control = result.safe_control.cwiseMin(
        Eigen::VectorXd::Constant(dimension, config.uav_acceleration_limit));
  }
  if (solve.success && config.method == Method::kMestres && !constraints.rows.empty())
    result.safe_control = projectedCorrection(result.safe_control, constraints, config, &result.distributed_rounds);
  // Python's oracle reports activity on the solver/projection control before
  // replacing an unsuccessful solve with its fail-safe command.  Preserve
  // that request-level diagnostic so failure cases remain comparable.
  const Eigen::VectorXd& active_control = solve.success ? result.safe_control : solve.control;
  result.active_constraints = 0;
  for (std::size_t index = 0; index < constraints.rows.size(); ++index)
    if (std::abs(constraints.rows[index].dot(active_control) - constraints.rhs[index]) < 1e-3)
      ++result.active_constraints;
  result.maximum_row_violation = maximumRowViolation(result.safe_control, constraints);
  result.maximum_bound_violation = maximumBoundViolation(result.safe_control, constraints);
  result.fallback = !solve.success;
  if (!solve.success) result.message = "CBF solver failed; fail-safe control applied";
  result.solve_time_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace hercules_cbf
