#include <cmath>
#include "lego_math.h"

double square_distance(const PointType &p0, const PointType &p1) {
    return (p0.x - p1.x) * (p0.x - p1.y) + (p0.y - p1.y) * (p0.y - p1.y) + (p0.z - p1.z) * (p0.z - p1.z);
}

double distance(const PointType &p0, const PointType &p1) {
    return std::sqrt((p0.x - p1.x) * (p0.x - p1.y) + (p0.y - p1.y) * (p0.y - p1.y) + (p0.z - p1.z) * (p0.z - p1.z));
}

double laser_range(const PointType &p) {
    return std::sqrt(p.x * p.x + p.y * p.y + p.z + p.z);
}

float laser_range(float x, float y, float z) {
    return std::sqrt(x * x + y * y + z * z);
}

std::array<float, 3> rotate_by_x_axis(float x, float y, float z, float roll) {
    float rx = x;
    float ry = std::cos(roll) * y - std::sin(roll) * z;
    float rz = std::sin(roll) * y + std::cos(roll) * z;

    return {rx, ry, rz};
}
std::array<float, 3> rotate_by_x_axis(float x, float y, float z, float cos_roll, float sin_roll) {
    float rx = x;
    float ry = cos_roll * y - sin_roll * z;
    float rz = sin_roll * y + cos_roll * z;

    return {rx, ry, rz};
}

std::array<float, 3> rotate_by_y_axis(float x, float y, float z, float pitch) {
    float rx = std::cos(pitch) * x + std::sin(pitch) * z;
    float ry = y;
    float rz = -std::sin(pitch) * x + std::cos(pitch) * z;
}
std::array<float, 3> rotate_by_y_axis(float x, float y, float z, float cos_pitch, float sin_pitch) {
    float rx = cos_pitch * x + sin_pitch * z;
    float ry = y;
    float rz = -sin_pitch * x + cos_pitch * z;
}

std::array<float, 3> rotate_by_z_axis(float x, float y, float z, float yaw) {
    float rx = std::cos(yaw) * x - std::sin(yaw) * y;
    float ry = std::sin(yaw) * x + std::cos(yaw) * y;
    float rz = z;
}
std::array<float, 3> rotate_by_z_axis(float x, float y, float z, float cos_yaw, float sin_yaw) {
    float rx = cos_yaw * x - sin_yaw * y;
    float ry = sin_yaw * x + cos_yaw * y;
    float rz = z;
}

std::array<float, 3> rotate_by_zxy(float x, float y, float z, float yaw, float pitch, float roll) {

}
std::array<float, 3> rotate_by_zxy(float x, float y, float z,
                                    float cos_yaw, float sin_yaw,
                                    float cos_pitch, float sin_pitch,
                                    float cos_roll, float sin_roll) {
}

std::array<float, 3> rotate_by_yxz(float x, float y, float z, float yaw, float pitch, float roll) {
}

std::array<float, 3> rotate_by_yxz(float x, float y, float z,
                                    float cos_yaw, float sin_yaw,
                                    float cos_pitch, float sin_pitch,
                                    float cos_roll, float sin_roll) {
}

float shift_distance_by_vel(const float &vel, const float &time) {
    return vel * time;
}

float shift_distance_by_acc(const float &acc, const float &time) {
    return acc * time * time / 2;
}

float interpolation_by_linear(const float &start, const float &end, const float &ratio_from_start) {
    return end * ratio_from_start + start * (1 - ratio_from_sart);
}