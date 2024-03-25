#ifndef LEGO_EIGEN_WRAPPER_H_
#define LEGO_EIGEN_WRAPPER_H_

#include <Eigen/Dense>
#include "types.h"

Point to_point(const Eigen::Vector3d &v);
Eigen::Vector3d to_vector(const Point &p);
Eigen::Matrix3d to_matrix(double roll, double pitch, double yaw);
Eigen::Quaterniond to_quaterniond(double roll, double pitch, double yaw);

Eigen::Vector3d rotate(const Point &p, double roll, double pitch, double yaw);
Eigen::Vector3d rotate_translate(const Point &p, const double *parameter);
Eigen::Vector3d rotate_translate(const Eigen::Vector3d &v, const double *parameter);

#endif
