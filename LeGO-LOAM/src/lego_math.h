// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#ifndef LEGO_MATH_H_
#define LEGO_MATH_H_

#include "utility.h"
#include <array>

#define PI 3.14159265

double rad2deg(double radian);
double deg2rad(double degree);

double square_distance(const Point &p0, const Point &p1);
double distance(const Point &p0, const Point &p1);
float laser_range(const Point &p0);
float laser_range(const float &x, const float &y, const float &z);

float shift_vel(const float &vel, const float &time);
float shift_acc(const float &acc, const float &time);

float interpolation_by_linear(const float &prev, const float &next, const float &ratio);

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

#endif  // LEGO_MATH_H_
