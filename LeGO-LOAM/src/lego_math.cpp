#include <cmath>
#include "lego_math.h"

double rad2deg(double radian) {
    return radian * 180.0 / M_PI;
}

double deg2rad(double degree) {
    return degree * M_PI / 180.0;
}

double square_distance(const Point &p0, const Point &p1) {
    return (p0.x - p1.x) * (p0.x - p1.y) + (p0.y - p1.y) * (p0.y - p1.y) + (p0.z - p1.z) * (p0.z - p1.z);
}

double distance(const Point &p0, const Point &p1) {
    return std::sqrt((p0.x - p1.x) * (p0.x - p1.y) + (p0.y - p1.y) * (p0.y - p1.y) + (p0.z - p1.z) * (p0.z - p1.z));
}

float laser_range(const Point &p) {
    return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

float laser_range(const float &x, const float &y, const float &z) {
    return std::sqrt(x * x + y * y + z * z);
}

std::array<float, 3> rotate_by_x_axis(const float &x, const float &y, const float &z, const float &roll) {
    float rx = x;
    float ry = std::cos(roll) * y - std::sin(roll) * z;
    float rz = std::sin(roll) * y + std::cos(roll) * z;

    return {rx, ry, rz};
}
std::array<float, 3> rotate_by_x_axis(const float &x, const float &y, const float &z, const float &cos_roll, const float &sin_roll) {
    float rx = x;
    float ry = cos_roll * y - sin_roll * z;
    float rz = sin_roll * y + cos_roll * z;

    return {rx, ry, rz};
}

std::array<float, 3> rotate_by_y_axis(const float &x, const float &y, const float &z, const float &pitch) {
    float rx = std::cos(pitch) * x + std::sin(pitch) * z;
    float ry = y;
    float rz = -std::sin(pitch) * x + std::cos(pitch) * z;

    return {rx, ry, rz};
}
std::array<float, 3> rotate_by_y_axis(const float &x, const float &y, const float &z, const float &cos_pitch, const float &sin_pitch) {
    float rx = cos_pitch * x + sin_pitch * z;
    float ry = y;
    float rz = -sin_pitch * x + cos_pitch * z;

    return {rx, ry, rz};
}

std::array<float, 3> rotate_by_z_axis(const float &x, const float &y, const float &z, const float &yaw) {
    float rx = std::cos(yaw) * x - std::sin(yaw) * y;
    float ry = std::sin(yaw) * x + std::cos(yaw) * y;
    float rz = z;

    return {rx, ry, rz};
}
std::array<float, 3> rotate_by_z_axis(const float &x, const float &y, const float &z, const float &cos_yaw, const float &sin_yaw) {
    float rx = cos_yaw * x - sin_yaw * y;
    float ry = sin_yaw * x + cos_yaw * y;
    float rz = z;

    return {rx, ry, rz};
}

std::array<float, 3> rotate_by_zxy(const float &x, const float &y, const float &z, const float &roll, const float &pitch, const float &yaw) {
    auto r0 = rotate_by_z_axis(x, y, z, yaw);
    auto r1 = rotate_by_x_axis(r0[0], r0[1], r0[2], roll);
    return rotate_by_y_axis(r1[0], r1[1], r1[2], pitch);
}
std::array<float, 3> rotate_by_zxy(const float &x, const float &y, const float &z,
                                    const float &cos_roll, const float &sin_roll,
                                    const float &cos_pitch, const float &sin_pitch,
                                    const float &cos_yaw, const float &sin_yaw) {
    auto r0 = rotate_by_z_axis(x, y, z, cos_yaw, sin_yaw);
    auto r1 = rotate_by_x_axis(r0[0], r0[1], r0[2], cos_roll, sin_roll);
    return rotate_by_y_axis(r1[0], r1[1], r1[2], cos_pitch, sin_pitch);
}

std::array<float, 3> rotate_by_yxz(const float &x, const float &y, const float &z, const float &yaw, const float &pitch, const float &roll) {
    auto r0 = rotate_by_y_axis(x, y, z, pitch);
    auto r1 = rotate_by_x_axis(r0[0], r0[1], r0[2], roll);
    return rotate_by_z_axis(r1[0], r1[1], r1[2], yaw);
}

std::array<float, 3> rotate_by_yxz(const float &x, const float &y, const float &z,
                                    const float &cos_roll, const float &sin_roll,
                                    const float &cos_pitch, const float &sin_pitch,
                                    const float &cos_yaw, const float &sin_yaw) {
    auto r0 = rotate_by_y_axis(x, y, z, cos_pitch, sin_pitch);
    auto r1 = rotate_by_x_axis(r0[0], r0[1], r0[2], cos_roll, sin_roll);
    return rotate_by_z_axis(r1[0], r1[1], r1[2], cos_yaw, sin_yaw);
}

float shift_vel(const float &vel, const float &time) {
    return vel * time;
}

float shift_acc(const float &acc, const float &time) {
    return acc * time * time / 2;
}

float interpolation_by_linear(const float &start, const float &end, const float &ratio_from_start) {
    return end * ratio_from_start + start * (1 - ratio_from_start);
}
