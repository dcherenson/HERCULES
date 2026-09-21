#pragma once

#include <Eigen/Core>

namespace hercules_localization {

/** Return the symmetric part of a square matrix. */
Eigen::MatrixXd symmetrize(const Eigen::Ref<const Eigen::MatrixXd>& matrix);

/** True only for finite, square matrices. */
bool isFiniteSquare(const Eigen::Ref<const Eigen::MatrixXd>& matrix);

/**
 * Make a covariance finite, symmetric, and positive definite.
 *
 * Non-finite entries are treated as zero, then eigenvalues are clamped to
 * `floor`.  The input dimension must be square and non-empty.  This routine
 * never returns a matrix with NaNs for a finite dimension/floor.
 */
Eigen::MatrixXd regularizeCovariance(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double floor = 1e-9);

/** A typed 2-D convenience wrapper around regularizeCovariance. */
Eigen::Matrix2d regularizeCovariance(const Eigen::Matrix2d& matrix,
                                     double floor = 1e-9);

/** A typed 3-D convenience wrapper around regularizeCovariance. */
Eigen::Matrix3d regularizeCovariance(const Eigen::Matrix3d& matrix,
                                     double floor = 1e-9);

/** Invert after positive-definite regularization. */
Eigen::MatrixXd safeInverse(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                            double floor = 1e-9);

Eigen::Matrix2d safeInverse(const Eigen::Matrix2d& matrix,
                            double floor = 1e-9);
Eigen::Matrix3d safeInverse(const Eigen::Matrix3d& matrix,
                            double floor = 1e-9);

/** Log determinant of the regularized covariance/information matrix. */
double logDeterminant(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                      double floor = 1e-9);

/** Symmetric Mahalanobis quadratic form with robust inversion. */
double mahalanobisSquared(const Eigen::Ref<const Eigen::VectorXd>& error,
                          const Eigen::Ref<const Eigen::MatrixXd>& covariance,
                          double floor = 1e-9);

/** Add isotropic variance for a standard-deviation robustness margin. */
Eigen::MatrixXd inflateCovariance(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                                  double standard_deviation,
                                  double floor = 1e-9);

}  // namespace hercules_localization
