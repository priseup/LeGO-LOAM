#include "eigen_wrapper.h"

Eigen::Vector3d to_vector(const Point &p)
{
  return {p.x, p.y, p.z};
}

Point to_point(const Eigen::Vector3d &v)
{
  Point p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

Eigen::Matrix3d to_matrix(double roll, double pitch, double yaw)
{
  Eigen::Matrix3d m;
  m = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
  return m;
}

Eigen::Quaterniond to_quaternion(double roll, double pitch, double yaw)
{
  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  return q;
}

Eigen::Vector3d rotate(const Point &p, double roll, double pitch, double yaw)
{
  return to_quaternion(roll, pitch, yaw) * to_vector(p);
}

Eigen::Vector3d rotate_translate(const Point &p, const double *parameter)
{
  return to_matrix(parameter[3], parameter[4], parameter[5]) * to_vector(p) + Eigen::Vector3d(parameter[0], parameter[1], parameter[2]);
}

Eigen::Vector3d rotate_translate(const Eigen::Vector3d &v, const double *parameter)
{
  return to_matrix(parameter[3], parameter[4], parameter[5]) * v + Eigen::Vector3d(parameter[0], parameter[1], parameter[2]);
}
