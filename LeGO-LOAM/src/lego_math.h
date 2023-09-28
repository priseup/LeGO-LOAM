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

#pragma once

#include <array>

double rad2deg(double radian) {
    return radian * 180.0 / M_PI;
}

double deg2rad(double degree) {
    return degree * M_PI / 180.0;
}

float time_liner_interpolation(float prev, float next, float ratio);

float laser_range(float x, float y, float z);

std::array<float, 3> rotate_by_zxy(float x, float y, float z, float yaw, float pitch, float roll);
std::array<float, 3> rotate_by_zxy(float x, float y, float z,
                                    float cos_yaw, float sin_yaw,
                                    float cos_pitch, float sin_pitch,
                                    float cos_roll, float sin_roll);
std::array<float, 3> rotate_by_yxz(float x, float y, float z, float yaw, float pitch, float roll);
std::array<float, 3> rotate_by_yxz(float x, float y, float z,
                                    float cos_yaw, float sin_yaw,
                                    float cos_pitch, float sin_pitch,
                                    float cos_roll, float sin_roll);

std::array<float, 3> rotate_by_x_axis(float x, float y, float z, float roll);
std::array<float, 3> rotate_by_x_axis(float x, float y, float z, float cos_roll, float sin_roll);

std::array<float, 3> rotate_by_y_axis(float x, float y, float z, float pitch);
std::array<float, 3> rotate_by_y_axis(float x, float y, float z, float cos_pitch, float sin_pitch);

std::array<float, 3> rotate_by_z_axis(float x, float y, float z, float yaw);
std::array<float, 3> rotate_by_z_axis(float x, float y, float z, float cos_yaw, float sin_yaw);
