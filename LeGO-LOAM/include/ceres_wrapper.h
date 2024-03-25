#ifndef LEGO_CERES_WRAPPER_H_
#define LEGO_CERES_WRAPPER_H_

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "types.h"
#include "utility.h"
#include "eigen_wrapper.h"

class CornerCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
  // CornerCostFunction(const Eigen::Vector3d &cp, const Eigen::Vector3d &lpj, const Eigen::Vector3d &lpl) : cp_(cp), lpj_(lpj), lpl_(lpl) {}
  CornerCostFunction(const Point &cp, const Point &lpj, const Point &lpl)
    : cp_(cp.x, cp.y, cp.z)
      , lpj_(lpj.x, lpj.y, lpj.z)
      , lpl_(lpl.x, lpl.y, lpl.z) {}
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
  {
    Eigen::Vector3d lp = rotate_translate(cp_, parameters[0]);
    // double a = (lp.y() - lpj_.y()) * (lp.z() - lpl_.z()) - (lp.z() - lpj_.z()) * (lp.y() - lpl_.y());
    // double b = (lp.z() - lpj_.z()) * (lp.x() - lpl_.x()) - (lp.x() - lpj_.x()) * (lp.z() - lpl_.z());
    // double c = (lp.x() - lpj_.x()) * (lp.y() - lpl_.y()) - (lp.y() - lpj_.y()) * (lp.x() - lpl_.x());
    auto coeff = (lpj_ - cp_).cross(lpl_ - cp_);
    double k = (lpl_ - lpj_).norm();
    residuals[0] = coeff.norm() / k;









    double a = (lp.x() - lpj_.x()) * (lp.y() - lpl_.y()) - (lp.x() - lpl_.x()) * (lp.y() - lpj_.y());
    // double b = (lp.x() - lpl_.x()) * (lp.z() - lpj_.z()) - (lp.x() - lpj_.x()) * (lp.z() - lpl_.z());
    double b = (lp.x() - lpj_.x()) * (lp.z() - lpl_.z()) - (lp.x() - lpl_.x()) * (lp.z() - lpj_.z());
    double c = (lp.y() - lpj_.y()) * (lp.z() - lpl_.z()) - (lp.y() - lpl_.y()) * (lp.z() - lpj_.z());
    double m = std::sqrt(a * a + b * b + c * c);

    residuals[0] = m / k;

    // double dm_dx = (b * (lpl_.z() - lpj_.z()) + c * (lpj_.y() - lpl_.y())) / m; --- error
    // double dm_dx = (b * (lpj_.z() - lpl_.z()) + c * (lpj_.y() - lpl_.y())) / m;
    // double dm_dy = (a * (lpj_.z() - lpl_.z()) - c * (lpj_.x() - lpl_.x())) / m;
    // double dm_dz = (-a * (lpj_.y() - lpl_.y()) + b * (lpj_.x() - lpl_.x())) / m;
    double dm_dx = (b * (lpj_.z() - lpl_.z()) + a * (lpj_.y() - lpl_.y())) / m / k;
    double dm_dy = (c * (lpj_.z() - lpl_.z()) - a * (lpj_.x() - lpl_.x())) / m / k;
    double dm_dz = -(c * (lpj_.y() - lpl_.y()) + b * (lpj_.x() - lpl_.x())) / m / k;

    double sr = std::sin(parameters[0][3]);
    double cr = std::cos(parameters[0][3]);
    double sp = std::sin(parameters[0][4]);
    double cp = std::cos(parameters[0][4]);
    double sy = std::sin(parameters[0][5]);
    double cy = std::cos(parameters[0][5]);

    double dx_dr = (cy * sp * cr + sr * sy) * cp_.y() + (sy * cr - cy * sr * sp) * cp_.z();
    double dy_dr = (-cy * sr + sy * sp * cr) * cp_.y() + (-sr * sy * sp - cy * cr) * cp_.z();
    double dz_dr = cp * cr * cp_.y() - cp * sr * cp_.z();

    double dx_dp = -cy * sp * cp_.x() + cy * cp * sr * cp_.y() + cy * cr * cp * cp_.z();
    double dy_dp = -sp * sy * cp_.x() + sy * cp * sr * cp_.y() + cr * sr * cp * cp_.z();
    double dz_dp = -cp * cp_.x() - sp * sr * cp_.y() - sp * cr * cp_.z();

    double dx_dy = -sy * cp * cp_.x() - (sy * sp * sr + cr * cy) * cp_.y() + (cy * sr - sy * cr * sp) * cp_.z();
    double dy_dy = cp * cy * cp_.x() + (-sy * cr + cy * sp * sr) * cp_.y() + (cy * cr * sp + sy * sr) * cp_.z();
    double dz_dy = 0.;

    if (jacobians && jacobians[0])
    {
      jacobians[0][0] = dm_dx / k;
      jacobians[0][1] = dm_dy / k;
      jacobians[0][2] = 0.;
      jacobians[0][3] = 0.;
      jacobians[0][4] = 0.;
      jacobians[0][5] = (dm_dx * dx_dy + dm_dy * dy_dy + dm_dz * dz_dy) / k;

      spdlog::info("\ncorner cost function");
      spdlog::info("residual: {}", residuals[0]);
      spdlog::info("J: {}, {}, {}\n", jacobians[0][0], jacobians[0][1], jacobians[0][5]);
    }

    return true;
  }

private:
  Eigen::Vector3d cp_;        // under t frame
  Eigen::Vector3d lpj_, lpl_; // under t-1 frame
};

class SurfCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
  // SurfCostFunction(const Eigen::Vector3d &cp, const Eigen::Vector3d &lpj, const Eigen::Vector3d &lpl, const Eigen::Vector3d &lpm) : cp_(cp), lpj_(lpj), lpl_(lpl), lpm_(lpm) {}
  SurfCostFunction(const Point &cp, const Point &lpj, const Point &lpl, const Point &lpm)
    : cp_(cp.x, cp.y, cp.z)
      , lpj_(lpj.x, lpj.y, lpj.z)
      , lpl_(lpl.x, lpl.y, lpl.z)
      , lpm_(lpm.x, lpm.y, lpm.z) {}
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
  {
    Eigen::Vector3d lp = rotate_translate(cp_, parameters[0]);
    // double a = (lpj_.y() - lpl_.y()) * (lpj_.z() - lpm_.z()) - (lpj_.z() - lpl_.z()) * (lpj_.y() - lpm_.y());
    // double b = (lpj_.z() - lpl_.z()) * (lpj_.x() - lpm_.x()) - (lpj_.x() - lpl_.x()) * (lpj_.z() - lpm_.z());
    // double c = (lpj_.x() - lpl_.x()) * (lpj_.y() - lpm_.y()) - (lpj_.y() - lpl_.y()) * (lpj_.x() - lpm_.x());
    
    // plane ax + by + cz + d = 0;
    // plane coeff.x() x + coeff.y() y + coeff.z() z + d = 0;
    auto coeff = (lpl_ - lpj_).cross(lpm_ - lpj_);
    double d = -(coeff.dot(lpj_));
    double k = coeff.norm();

    residuals[0] = (coeff.dot(cp_) + d) / k;

    double m = std::sqrt(std::pow((lp.x() - lpj_.x()), 2) * a + std::pow((lp.y() - lpj_.y()), 2) * b + std::pow((lp.z() - lpj_.z()), 2) * c);
    double tmp = m * k;

    double dm_dx = ((lp.x() - lpj_.x()) * a) / tmp;
    double dm_dy = ((lp.y() - lpj_.y()) * b) / tmp;
    double dm_dz = ((lp.z() - lpj_.z()) * c) / tmp;

    double sr = std::sin(parameters[0][3]);
    double cr = std::cos(parameters[0][3]);
    double sp = std::sin(parameters[0][4]);
    double cp = std::cos(parameters[0][4]);
    double sy = std::sin(parameters[0][5]);
    double cy = std::cos(parameters[0][5]);

    double dx_dr = (cy * sp * cr + sr * sy) * cp_.y() + (sy * cr - cy * sr * sp) * cp_.z();
    double dy_dr = (-cy * sr + sy * sp * cr) * cp_.y() + (-sr * sy * sp - cy * cr) * cp_.z();
    double dz_dr = cp * cr * cp_.y() - cp * sr * cp_.z();

    double dx_dp = -cy * sp * cp_.x() + cy * cp * sr * cp_.y() + cy * cr * cp * cp_.z();
    double dy_dp = -sp * sy * cp_.x() + sy * cp * sr * cp_.y() + cr * sr * cp * cp_.z();
    double dz_dp = -cp * cp_.x() - sp * sr * cp_.y() - sp * cr * cp_.z();

    double dx_dy = -sy * cp * cp_.x() - (sy * sp * sr + cr * cy) * cp_.y() + (cy * sr - sy * cr * sp) * cp_.z();
    double dy_dy = cp * cy * cp_.x() + (-sy * cr + cy * sp * sr) * cp_.y() + (cy * cr * sp + sy * sr) * cp_.z();
    double dz_dy = 0.;

    if (jacobians && jacobians[0])
    {
      jacobians[0][0] = 0.;
      jacobians[0][1] = 0.;
      jacobians[0][2] = dm_dz / k;
      jacobians[0][3] = 0.;
      jacobians[0][4] = 0.;
      jacobians[0][5] = 0.;

      spdlog::info("surf cost function");
      spdlog::info("residual: {}", residuals[0]);
      spdlog::info("J: {}", jacobians[0][2]);
    }

    return true;
}
  
private:
  Eigen::Vector3d cp_;
  Eigen::Vector3d lpj_, lpl_, lpm_;
};

class LidarEdgeCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
  LidarEdgeCostFunction(const Eigen::Vector3d &cp, const Eigen::Vector3d &lpj, const Eigen::Vector3d &lpl) : cp_(cp), lpj_(lpj), lpl_(lpl) {}
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
  {
    Eigen::Vector3d lp = rotate_translate(cp_, parameters[0]);
    double a = (lp.y() - lpj_.y()) * (lp.z() - lpl_.z()) - (lp.z() - lpj_.z()) * (lp.y() - lpl_.y());
    double b = (lp.z() - lpj_.z()) * (lp.x() - lpl_.x()) - (lp.x() - lpj_.x()) * (lp.z() - lpl_.z());
    double c = (lp.x() - lpj_.x()) * (lp.y() - lpl_.y()) - (lp.y() - lpj_.y()) * (lp.x() - lpl_.x());
    double m = std::sqrt(a * a + b * b + c * c);
    double k = (lpj_ - lpl_).norm();

    residuals[0] = m / k;

    double dm_dx = (b * (lpj_.z() - lpl_.z()) + c * (lpj_.y() - lpl_.y())) / m;
    double dm_dy = (a * (lpj_.z() - lpl_.z()) - c * (lpj_.x() - lpl_.x())) / m;
    double dm_dz = (-a * (lpj_.y() - lpl_.y()) + b * (lpj_.x() - lpl_.x())) / m;

    double sr = std::sin(parameters[0][3]);
    double cr = std::cos(parameters[0][3]);
    double sp = std::sin(parameters[0][4]);
    double cp = std::cos(parameters[0][4]);
    double sy = std::sin(parameters[0][5]);
    double cy = std::cos(parameters[0][5]);

    double dx_dr = (cy * sp * cr + sr * sy) * cp_.y() + (sy * cr - cy * sr * sp) * cp_.z();
    double dy_dr = (-cy * sr + sy * sp * cr) * cp_.y() + (-sr * sy * sp - cy * cr) * cp_.z();
    double dz_dr = cp * cr * cp_.y() - cp * sr * cp_.z();

    double dx_dp = -cy * sp * cp_.x() + cy * cp * sr * cp_.y() + cy * cr * cp * cp_.z();
    double dy_dp = -sp * sy * cp_.x() + sy * cp * sr * cp_.y() + cr * sr * cp * cp_.z();
    double dz_dp = -cp * cp_.x() - sp * sr * cp_.y() - sp * cr * cp_.z();

    double dx_dy = -sy * cp * cp_.x() - (sy * sp * sr + cr * cy) * cp_.y() + (cy * sr - sy * cr * sp) * cp_.z();
    double dy_dy = cp * cy * cp_.x() + (-sy * cr + cy * sp * sr) * cp_.y() + (cy * cr * sp + sy * sr) * cp_.z();
    double dz_dy = 0.;

    if (jacobians && jacobians[0])
    {
      jacobians[0][0] = dm_dx / k;
      jacobians[0][1] = dm_dy / k;
      jacobians[0][2] = dm_dz / k;
      jacobians[0][3] = (dm_dx * dx_dr + dm_dy * dy_dr + dm_dz * dz_dr) / k;
      jacobians[0][4] = (dm_dx * dx_dp + dm_dy * dy_dp + dm_dz * dz_dp) / k;
      jacobians[0][5] = (dm_dx * dx_dy + dm_dy * dy_dy + dm_dz * dz_dy) / k;
    }

    return true;
  }

  private:
    Eigen::Vector3d cp_;        // under t frame
    Eigen::Vector3d lpj_, lpl_; // under t-1 frame
};

class LidarPlaneCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
  LidarPlaneCostFunction(const Eigen::Vector3d &cp, const Eigen::Vector3d &plane_unit_norm,
                         double negative_OA_dot_norm) : cp_(cp), plane_unit_norm_(plane_unit_norm),
                                                        negative_OA_dot_norm_(negative_OA_dot_norm) {}
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
  {
    Eigen::Vector3d lp = rotate_translate(cp_, parameters[0]);
    residuals[0] = plane_unit_norm_.dot(lp) + negative_OA_dot_norm_;
    Eigen::Vector3d df_dxyz = plane_unit_norm_;

    double sr = std::sin(parameters[0][3]);
    double cr = std::cos(parameters[0][3]);
    double sp = std::sin(parameters[0][4]);
    double cp = std::cos(parameters[0][4]);
    double sy = std::sin(parameters[0][5]);
    double cy = std::cos(parameters[0][5]);

    double dx_dr = (cy * sp * cr + sr * sy) * cp_.y() + (sy * cr - cy * sr * sp) * cp_.z();
    double dy_dr = (-cy * sr + sy * sp * cr) * cp_.y() + (-sr * sy * sp - cy * cr) * cp_.z();
    double dz_dr = cp * cr * cp_.y() - cp * sr * cp_.z();

    double dx_dp = -cy * sp * cp_.x() + cy * cp * sr * cp_.y() + cy * cr * cp * cp_.z();
    double dy_dp = -sp * sy * cp_.x() + sy * cp * sr * cp_.y() + cr * sr * cp * cp_.z();
    double dz_dp = -cp * cp_.x() - sp * sr * cp_.y() - sp * cr * cp_.z();

    double dx_dy = -sy * cp * cp_.x() - (sy * sp * sr + cr * cy) * cp_.y() + (cy * sr - sy * cr * sp) * cp_.z();
    double dy_dy = cp * cy * cp_.x() + (-sy * cr + cy * sp * sr) * cp_.y() + (cy * cr * sp + sy * sr) * cp_.z();
    double dz_dy = 0.;

    if (jacobians && jacobians[0])
    {
      jacobians[0][0] = df_dxyz.x();
      jacobians[0][1] = df_dxyz.y();
      jacobians[0][2] = df_dxyz.z();
      jacobians[0][3] = df_dxyz.x() * dx_dr + df_dxyz.y() * dy_dr + df_dxyz.z() * dz_dr;
      jacobians[0][4] = df_dxyz.x() * dx_dp + df_dxyz.y() * dy_dp + df_dxyz.z() * dz_dp;
      jacobians[0][5] = df_dxyz.x() * dx_dy + df_dxyz.y() * dy_dy + df_dxyz.z() * dz_dy;
    }

    return true;
}

private:
  Eigen::Vector3d cp_;
  Eigen::Vector3d plane_unit_norm_;
  double negative_OA_dot_norm_;
};

void solve_problem(ceres::Problem &problem);

#endif
