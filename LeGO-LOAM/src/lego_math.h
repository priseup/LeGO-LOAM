#ifndef LEGO_MATH_H_
#define LEGO_MATH_H_

#include "types.h"

#define PI 3.14159265

double rad2deg(double radian);
double deg2rad(double degree);

double square_distance(const Point &p0, const Point &p1);
double distance(const Point &p0, const Point &p1);
float laser_range(const Point &p0);
float laser_range(const float &x, const float &y, const float &z);

float interpolation_by_linear(const float &prev, const float &next, const float &ratio);

/*
std::array<float, 3> rotate_by_x_axis(const float &x, const float &y, const float &z, const float &roll);
std::array<float, 3> rotate_by_x_axis(const float &x, const float &y, const float &z, const float &cos_roll, const float &sin_roll);

std::array<float, 3> rotate_by_y_axis(const float &x, const float &y, const float &z, const float &pitch);
std::array<float, 3> rotate_by_y_axis(const float &x, const float &y, const float &z, const float &cos_pitch, const float &sin_pitch);

std::array<float, 3> rotate_by_z_axis(const float &x, const float &y, const float &z, const float &yaw);
std::array<float, 3> rotate_by_z_axis(const float &x, const float &y, const float &z, const float &cos_yaw, const float &sin_yaw);

std::array<float, 3> rotate_by_zxy(const float &x, const float &y, const float &z,
                                    const float &roll, const float &pitch, const float &yaw);
std::array<float, 3> rotate_by_zxy(const float &x, const float &y, const float &z,
                                const float &cos_roll, const float &sin_roll,
                                const float &cos_pitch, const float &sin_pitch,
                                const float &cos_yaw, const float &sin_yaw);

std::array<float, 3> rotate_by_yxz(const float &x, const float &y, const float &z, const float &roll, const float &pitch, const float &yaw);
std::array<float, 3> rotate_by_yxz(const float &x, const float &y, const float &z,
                                const float &cos_roll, const float &sin_roll,
                                const float &cos_pitch, const float &sin_pitch,
                                const float &cos_yaw, const float &sin_yaw);

*/

#endif  // LEGO_MATH_H_
