#include <cmath>
#include <algorithm>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "utility.h"
#include "laser_info.h"
#include "lego_math.h"
#include "eigen_wrapper.h"
#include "ceres_wrapper.h"
#include "featureAssociation.h"

static int feature_idx_ = 0;

const int edgeFeatureNum = 2;
const int surfFeatureNum = 4;
const float edgeThreshold = 0.1;
const float surfThreshold = 0.1;
const float nearestFeatureSearchSqDist = 25;

FeatureAssociation::FeatureAssociation(): nh_("~") {
    sub_laser_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>("/ground_with_segmented_cloud", 1, &FeatureAssociation::laser_cloud_handler, this);
    sub_laser_cloud_info_ = nh_.subscribe<cloud_msgs::cloud_info>("/ground_with_segmented_cloud_info", 1, &FeatureAssociation::laser_cloud_msg_handler, this);
    sub_outlier_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &FeatureAssociation::outlier_cloud_handler, this);
    sub_imu_ = nh_.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureAssociation::imu_handler, this);

    pub_corner_sharp_ = nh_.advertise<sensor_msgs::PointCloud2>("/corner_sharp", 1);
    pub_corner_less_sharp_ = nh_.advertise<sensor_msgs::PointCloud2>("/corner_less_sharp", 1);
    pub_surf_flat_ = nh_.advertise<sensor_msgs::PointCloud2>("/surf_flat", 1);
    pub_surf_less_flat_ = nh_.advertise<sensor_msgs::PointCloud2>("/surf_less_flat", 1);

    pub_last_corner_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/corner_last", 2);
    pub_last_surf_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/surf_last", 2);
    pub_last_outlier_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/outlier_last", 2);
    pub_laser_odometry_ = nh_.advertise<nav_msgs::Odometry> ("/laser_odom", 5);

    cloud_last_corner_.reset(new pcl::PointCloud<Point>);
    cloud_last_surf_.reset(new pcl::PointCloud<Point>);

    kd_last_corner_.reset(new pcl::KdTreeFLANN<Point>);
    kd_last_surf_.reset(new pcl::KdTreeFLANN<Point>);

    projected_ground_segment_cloud_.reset(new pcl::PointCloud<Point>);
    projected_outlier_cloud_.reset(new pcl::PointCloud<Point>);

    corner_sharp_cloud_.reset(new pcl::PointCloud<Point>);
    corner_less_sharp_cloud_.reset(new pcl::PointCloud<Point>);
    surf_flat_cloud_.reset(new pcl::PointCloud<Point>);
    surf_less_flat_cloud_.reset(new pcl::PointCloud<Point>);

    r_w_curr_.setIdentity();
    t_w_curr_.setZero();

    init();
}
void FeatureAssociation::reset_parameters() {
    corner_sharp_cloud_->clear();
    corner_less_sharp_cloud_->clear();
    surf_flat_cloud_->clear();
    surf_less_flat_cloud_->clear();

    std::fill(is_neibor_picked_.begin(), is_neibor_picked_.end(), 0);
}

void FeatureAssociation::init()
{
    cloud_smoothness_.resize(points_num);
    cloud_curvature_.resize(points_num);
    is_neibor_picked_.resize(points_num);
    cloud_label_.resize(points_num, FeatureLabel::surf_less_flat);
}

void FeatureAssociation::imu_handler(const sensor_msgs::Imu::ConstPtr &imu)
{
    const auto &imu_last_new = imus_[imu_idx_new_];

    imu_idx_new_ = (imu_idx_new_ + 1) % imuQueLength; // first imu newer than laser point time
    auto &imu_new = imus_[imu_idx_new_];
    imu_new.time = imu->header.stamp.toSec();

    double roll, pitch, yaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(imu->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    imu_new.roll = (float)roll;
    imu_new.pitch = (float)pitch;
    imu_new.yaw = (float)yaw;

    imu_new.acc_x = imu->linear_acceleration.x + std::sin(pitch) * 9.81;
    imu_new.acc_y = imu->linear_acceleration.y - std::sin(roll) * std::cos(pitch) * 9.81;
    imu_new.acc_z = imu->linear_acceleration.z - std::cos(roll) * std::cos(pitch) * 9.81;

    imu_new.angular_vel_x = imu->angular_velocity.x;
    imu_new.angular_vel_y = imu->angular_velocity.y;
    imu_new.angular_vel_z = imu->angular_velocity.z;

    double time_diff = imu_new.time - imu_last_new.time;
    if (time_diff < scanPeriod) {
        Eigen::Vector3f acc = Eigen::Quaternionf(imu->orientation.w, imu->orientation.x, imu->orientation.y, imu->orientation.z) * Eigen::Vector3f(imu_new.acc_x, imu_new.acc_y, imu_new.acc_z);

        imu_new.vel_x = imu_last_new.vel_x + acc[0] * time_diff;
        imu_new.vel_y = imu_last_new.vel_y + acc[1] * time_diff;
        imu_new.vel_z = imu_last_new.vel_z + acc[2] * time_diff;

        imu_new.shift_x = imu_last_new.shift_x + imu_last_new.vel_x * time_diff + acc[0] * time_diff * time_diff * 0.5;
        imu_new.shift_y = imu_last_new.shift_y + imu_last_new.vel_y * time_diff + acc[1] * time_diff * time_diff * 0.5;
        imu_new.shift_z = imu_last_new.shift_z + imu_last_new.vel_z * time_diff + acc[2] * time_diff * time_diff * 0.5;

        imu_new.angular_x = imu_last_new.angular_x + imu_last_new.angular_vel_x * time_diff;
        imu_new.angular_y = imu_last_new.angular_y + imu_last_new.angular_vel_y * time_diff;
        imu_new.angular_z = imu_last_new.angular_z + imu_last_new.angular_vel_z * time_diff;
    }
}

void FeatureAssociation::laser_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& laser_cloud) {
    cloud_header_ = laser_cloud->header;

    laser_scan_time_ = cloud_header_.stamp.toSec();
    segment_cloud_time_ = laser_scan_time_;

    pcl::fromROSMsg(*laser_cloud, *projected_ground_segment_cloud_);

    has_get_cloud_ = true;

/*
    pcl::PointCloud<Point> tmp_save_cloud;
    for (auto &p : projected_ground_segment_cloud_->points) {
      Point ap;
      ap.x = p.x;
      ap.y = p.y;
      ap.z = p.z;
      ap.intensity = p.intensity;
      tmp_save_cloud.push_back(ap);
    }
    pcl::io::savePCDFileASCII(fmt::format("/home/pqf/my_lego/{}_raw_cloud.pcd", feature_idx_), tmp_save_cloud);
*/
}

void FeatureAssociation::outlier_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& msgIn) {
    outlier_cloud_time_ = msgIn->header.stamp.toSec();

    pcl::fromROSMsg(*msgIn, *projected_outlier_cloud_);

    has_get_outlier_cloud_ = true;
}

void FeatureAssociation::laser_cloud_msg_handler(const cloud_msgs::cloud_infoConstPtr& msgIn) {
    segment_cloud_info_time_ = msgIn->header.stamp.toSec();
    segmented_cloud_msg_ = *msgIn;

    has_get_cloud_msg_ = true;
}

void FeatureAssociation::adjust_distortion() {
    bool is_half_pass = false;

    for (int i = 0; i < projected_ground_segment_cloud_->points.size(); i++) {
        auto &point = projected_ground_segment_cloud_->points[i];
        float horizontal_angle = -std::atan2(point.y, point.x);     // why -atan2

        if (!is_half_pass) {
            if (horizontal_angle < segmented_cloud_msg_.orientation_start - M_PI / 2)
                horizontal_angle += 2 * M_PI;
            else if (horizontal_angle > segmented_cloud_msg_.orientation_start + M_PI * 3 / 2)
                horizontal_angle -= 2 * M_PI;

            if (horizontal_angle - segmented_cloud_msg_.orientation_start > M_PI)
                is_half_pass = true;
        } else {
            horizontal_angle += 2 * M_PI;

            if (horizontal_angle < segmented_cloud_msg_.orientation_end - M_PI * 3 / 2)
                horizontal_angle += 2 * M_PI;
            else if (horizontal_angle > segmented_cloud_msg_.orientation_end + M_PI / 2)
                horizontal_angle -= 2 * M_PI;
        }

        float point_ratio = (horizontal_angle - segmented_cloud_msg_.orientation_start) / segmented_cloud_msg_.orientation_diff;
        point.intensity += point_ratio;
        float point_time = point_ratio * scanPeriod;
        float laser_point_time = laser_scan_time_ + point_time;

        Eigen::Vector3f rpy_start, rpy_cur;
        Eigen::Vector3f vel_start, vel_cur;
        Eigen::Vector3f shift_start, shift_cur;
        Eigen::Vector3f shift_from_start;
        Eigen::Matrix3f r_s_i, r_c;

        if (imu_idx_new_ >= 0) {
            imu_idx_after_laser_ = imu_idx_last_used_;
            while (laser_point_time < imus_[imu_idx_after_laser_].time) {
                if (imu_idx_after_laser_ != imu_idx_new_) {
                    break;
                }
                imu_idx_after_laser_ = (imu_idx_after_laser_ + 1) % imuQueLength;
            }

            const auto &imu_after_laser = imus_[imu_idx_after_laser_];
            if (laser_point_time > imu_after_laser.time) {
                // imu_idx_new_ == imu_idx_after_laser_ in this case
                rpy_cur[0] = imu_after_laser.roll;
                rpy_cur[1] = imu_after_laser.pitch;
                rpy_cur[2] = imu_after_laser.yaw;

                vel_cur[0] = imu_after_laser.vel_x;
                vel_cur[1] = imu_after_laser.vel_y;
                vel_cur[2] = imu_after_laser.vel_z;

                shift_cur[0] = imu_after_laser.shift_x;
                shift_cur[1] = imu_after_laser.shift_y;
                shift_cur[2] = imu_after_laser.shift_z;
            } else {
                int idx_before_laser = (imu_idx_after_laser_ + imuQueLength - 1) % imuQueLength;
                const auto &imu_before_laser = imus_[idx_before_laser];
                float ratio_from_start = (laser_point_time - imu_before_laser.time) 
                                        / (imu_after_laser.time - imu_before_laser.time);

                float imu_before_yaw = imu_before_laser.yaw;
                if (imu_after_laser.yaw - imu_before_laser.yaw > M_PI) {
                    imu_before_yaw += 2 * M_PI;
                } else if (imu_after_laser.yaw - imu_before_laser.yaw < -M_PI) {
                    imu_before_yaw -= 2 * M_PI;
                }
                rpy_cur[0] = interpolation_by_linear(imu_before_laser.roll, imu_after_laser.roll, ratio_from_start);
                rpy_cur[1] = interpolation_by_linear(imu_before_laser.pitch, imu_after_laser.pitch, ratio_from_start);
                rpy_cur[2] = interpolation_by_linear(imu_before_yaw, imu_after_laser.yaw, ratio_from_start);

                vel_cur[0] = interpolation_by_linear(imu_before_laser.vel_x, imu_after_laser.vel_x, ratio_from_start);
                vel_cur[1] = interpolation_by_linear(imu_before_laser.vel_y, imu_after_laser.vel_y, ratio_from_start);
                vel_cur[2] = interpolation_by_linear(imu_before_laser.vel_z, imu_after_laser.vel_z, ratio_from_start);

                shift_cur[0] = interpolation_by_linear(imu_before_laser.shift_x, imu_after_laser.shift_x, ratio_from_start);
                shift_cur[1] = interpolation_by_linear(imu_before_laser.shift_y, imu_after_laser.shift_y, ratio_from_start);
                shift_cur[2] = interpolation_by_linear(imu_before_laser.shift_z, imu_after_laser.shift_z, ratio_from_start);
            }

            r_c = (Eigen::AngleAxisf(rpy_cur[2], Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(rpy_cur[1], Eigen::Vector3f::UnitY()) * Eigen::AngleAxisf(rpy_cur[0], Eigen::Vector3f::UnitX())).toRotationMatrix();
            if (i == 0) {
                rpy_start = rpy_cur;
                vel_start = vel_cur;
                shift_start = shift_cur;

                r_s_i = r_c.inverse();
            } else {
                // vel_to_start_imu();
                // transform_to_start_imu(point);
                shift_from_start = shift_cur - shift_start - vel_start * point_time;
                auto adjusted_p = r_s_i * (r_c * Eigen::Vector3f{point.x, point.y, point.z} + shift_from_start);
                point.x = adjusted_p[0];
                point.y = adjusted_p[1];
                point.z = adjusted_p[2];
            }
        }
    }
    imu_idx_last_used_ = imu_idx_after_laser_;

    pcl::PointCloud<Point> tmp_save_cloud;
    for (auto &p : projected_ground_segment_cloud_->points) {
      Point ap;
      ap.x = p.x;
      ap.y = p.y;
      ap.z = p.z;
      ap.intensity = p.intensity;
      tmp_save_cloud.push_back(ap);
    }
    pcl::io::savePCDFileASCII(fmt::format("/home/pqf/my_lego/{}_adjust_distortion.pcd", feature_idx_), tmp_save_cloud);
}

void FeatureAssociation::calculate_smoothness()
{
    for (int i = 5; i < (int)projected_ground_segment_cloud_->points.size() - 5; i++) {
        const auto &cloud_range = segmented_cloud_msg_.ground_segment_cloud_range; 
        float diff_range = cloud_range[i-5] + cloud_range[i-4]
                        + cloud_range[i-3] + cloud_range[i-2]
                        + cloud_range[i-1] - cloud_range[i] * 10
                        + cloud_range[i+1] + cloud_range[i+2]
                        + cloud_range[i+3] + cloud_range[i+4]
                        + cloud_range[i+5];            

        cloud_curvature_[i] = diff_range * diff_range;

        cloud_smoothness_[i] = {cloud_curvature_[i], i};
    }
}

void FeatureAssociation::mark_occluded_points()
{
    const auto &cloud_range = segmented_cloud_msg_.ground_segment_cloud_range;
    const auto &cloud_column = segmented_cloud_msg_.ground_segment_cloud_column;

    for (int i = 5; i < projected_ground_segment_cloud_->points.size() - 6; ++i) {
        float range_i = cloud_range[i];
        float range_i_1 = cloud_range[i+1];

        if (std::abs(int(cloud_column[i+1] - cloud_column[i])) < 10) {
            if (range_i - range_i_1 > 0.3) {
                std::fill(is_neibor_picked_.begin() + i - 5, is_neibor_picked_.begin() + i + 1, 1);
            }else if (range_i_1 - range_i > 0.3) {
                std::fill(is_neibor_picked_.begin() + i + 1, is_neibor_picked_.begin() + i + 7, 1);
            }
        }

        float diff_prev = std::abs(cloud_range[i-1] - cloud_range[i]);
        float diff_next = std::abs(cloud_range[i+1] - cloud_range[i]);

        if (diff_prev > 0.02 * cloud_range[i] && diff_next > 0.02 * cloud_range[i])
            is_neibor_picked_[i] = 1;
    }
}

void FeatureAssociation::mark_neibor_is_picked(int idx)
{
    const auto &cloud_column = segmented_cloud_msg_.ground_segment_cloud_column;
    for (int i = 1; i <= 5; i++) {
        if (std::abs(int(cloud_column[idx + i] - cloud_column[idx + i - 1])) > 10)
            break;
        is_neibor_picked_[idx + i] = 1;
    }
    for (int i = -1; i >= -5; i--) {
        if (std::abs(int(cloud_column[idx + i] - cloud_column[idx + i + 1])) > 10)
            break;
        is_neibor_picked_[idx + i] = 1;
    }
}

void FeatureAssociation::extract_features()
{
    const auto &cloud = projected_ground_segment_cloud_->points;
    const auto &cloud_range = segmented_cloud_msg_.ground_segment_cloud_range;
    const auto &cloud_column = segmented_cloud_msg_.ground_segment_cloud_column;

    for (int i = 0; i < N_SCAN; i++) {
      int scan_start = segmented_cloud_msg_.ring_index_start[i];
      int scan_end = segmented_cloud_msg_.ring_index_end[i];
        for (int j = 0; j < 6; j++) {
            int sp = std::max((scan_start * (6 - j)  + scan_end * j) / 6, 5);
            int ep = (scan_start * (5 - j) + scan_end * (j + 1)) / 6 - 1;
            if (sp >= ep)
                continue;
            std::sort(cloud_smoothness_.begin() + sp, cloud_smoothness_.begin() + ep + 1);

            int pick_num = 0;
            for (int k = ep; k >= sp; k--) {
                int idx = cloud_smoothness_[k].idx;
        
                if (is_neibor_picked_[idx] == 0
                    && !segmented_cloud_msg_.ground_segment_flag[idx]
                    && cloud_curvature_[idx] > edgeThreshold) {
                    pick_num++;
                    if (pick_num <= edgeFeatureNum) {
                        cloud_label_[idx] = FeatureLabel::corner_sharp;
                        corner_sharp_cloud_->push_back(cloud[idx]);
                        corner_less_sharp_cloud_->push_back(cloud[idx]);
                    } else if (pick_num <= 20) {
                        cloud_label_[idx] = FeatureLabel::corner_less_sharp;
                        corner_less_sharp_cloud_->push_back(cloud[idx]);
                    } else {
                        break;
                    }

                    is_neibor_picked_[idx] = 1;
                    mark_neibor_is_picked(idx);
                }
            }

            pick_num = 0;
            for (int k = sp; k <= ep; k++) {
                int idx = cloud_smoothness_[k].idx;

                if (is_neibor_picked_[idx] == 0
                    && segmented_cloud_msg_.ground_segment_flag[idx]
                    && cloud_curvature_[idx] < surfThreshold) {
                    cloud_label_[idx] = FeatureLabel::surf_flat;
                    surf_flat_cloud_->push_back(cloud[idx]);
                    surf_less_flat_cloud_->push_back(cloud[idx]);
                    if (pick_num++ >= surfFeatureNum) {
                        break;
                    }

                    is_neibor_picked_[idx] = 1;
                    mark_neibor_is_picked(idx);
                }
            }

            for (int k = sp; k <= ep; k++) {
                if (cloud_label_[k] == FeatureLabel::surf_less_flat) {
                    surf_less_flat_cloud_->push_back(cloud[k]);
                }
            }
        }
    }
    /*
    static pcl::PointCloud<Point>::Ptr tmp_cloud(new pcl::PointCloud<Point>);
    static pcl::VoxelGrid<Point> ds_less_flat;
    ds_less_flat.setInputCloud(surf_less_flat_cloud_);
    ds_less_flat.setLeafSize(0.2, 0.2, 0.2);
    ds_less_flat.filter(*tmp_cloud);
    spdlog::info("less_suf, after filter: {}, {}", surf_less_flat_cloud_->size(), tmp_cloud->size());
    *surf_less_flat_cloud_ = *tmp_cloud;
    tmp_cloud->clear();
    */

    static auto feature_log = spdlog::basic_logger_st("feature", "/home/pqf/my_lego/feature_log.txt", true);
    feature_log->info("sharp, less_sharp, surf, less_surf: {}, {}, {}, {}", corner_sharp_cloud_->size(), corner_less_sharp_cloud_->size(), surf_flat_cloud_->size(), surf_less_flat_cloud_->size());
    feature_log->flush();

/*
    pcl::PointCloud<Point> tmp_save_cloud;
    for (auto &p : surf_less_flat_cloud_->points) {
      Point ap;
      ap.x = p.x;
      ap.y = p.y;
      ap.z = p.z;
      ap.intensity = p.intensity;
      tmp_save_cloud.push_back(ap);
    }
    pcl::io::savePCDFileASCII(fmt::format("/home/pqf/my_lego/{}_surf_less.pcd", feature_idx_), tmp_save_cloud);
    feature_idx_++;

    tmp_save_cloud.clear();
*/
}

void FeatureAssociation::publish_cloud()
{
    sensor_msgs::PointCloud2 laser_cloud_msg;

    if (pub_corner_sharp_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*corner_sharp_cloud_, laser_cloud_msg);
        laser_cloud_msg.header = cloud_header_;
        pub_corner_sharp_.publish(laser_cloud_msg);
    }

    if (pub_corner_less_sharp_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*corner_less_sharp_cloud_, laser_cloud_msg);
        laser_cloud_msg.header = cloud_header_;
        pub_corner_less_sharp_.publish(laser_cloud_msg);
    }

    if (pub_surf_flat_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*surf_flat_cloud_, laser_cloud_msg);
        laser_cloud_msg.header = cloud_header_;
        pub_surf_flat_.publish(laser_cloud_msg);
    }

    if (pub_surf_less_flat_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*surf_less_flat_cloud_, laser_cloud_msg);
        laser_cloud_msg.header = cloud_header_;
        pub_surf_less_flat_.publish(laser_cloud_msg);
    }
}

Point FeatureAssociation::transform_to_start(const Point &p)
{
    float s = p.intensity - int(p.intensity);
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr_);
    Eigen::Vector3d t_point_last = s * t_last_curr_;

    Eigen::Vector3d sp = q_point_last * to_vector(p) + t_point_last;

    Point po;
    po.x = sp[0];
    po.y = sp[1];
    po.z = sp[2];
    po.intensity = p.intensity;

    return po;
}

void FeatureAssociation::transform_to_end(Point &p)
{
    auto v = q_last_curr_.inverse() * (to_vector(transform_to_start(p)) - t_last_curr_);

    p.x = v[0];
    p.y = v[1];
    p.z = v[2];
    p.intensity = (int)p.intensity;
}

int FeatureAssociation::point_scan_id(const Point &p) {
    return int(p.intensity);
}

std::array<int, 2> FeatureAssociation::find_closest_in_same_adjacent_ring(int closest_idx, const Point &p, const pcl::PointCloud<Point>::Ptr &cloud, bool get_same) {
    int min_idx_same = -1;
    int min_idx_adj = -1;
    float min_dis_same = nearestFeatureSearchSqDist;
    float min_dis_adj = nearestFeatureSearchSqDist;
    int closest_scan_id = point_scan_id(cloud->points[closest_idx]);

/*
    for (int i = closest_idx + 1; i < cloud->points.size(); i++) {
        if (point_scan_id(cloud->points[i]) > closest_scan_id + 3) {
            break; 
        }

        float d = square_distance(cloud->points[i], p);
        if (get_same && point_scan_id(cloud->points[i]) <= closest_scan_id) {
            if (d < min_dis_same) {
                min_dis_same = d;
                min_idx_same = i;
            }
        } else {
            if (d < min_dis_adj) {
                min_dis_adj = d;
                min_idx_adj = i;
            }
        }
    }
    for (int i = closest_idx - 1; i >= 0; i--) {
        if (point_scan_id(cloud->points[i]) < closest_scan_id - 3) {
            break;
        }

        float d = square_distance(cloud->points[i], p);
        if (get_same && point_scan_id(cloud->points[i]) >= closest_scan_id) {
            if (d < min_dis_same) {
                min_dis_same = d;
                min_idx_same = i;
            }
        } else {
            if (d < min_dis_adj) {
                min_dis_adj = d;
                min_idx_adj = i;
            }
        }
    }

    return {min_idx_same, min_idx_adj};
*/

    for (int i = segmented_cloud_msg_.ring_index_start[closest_scan_id - 1]; i <= segmented_cloud_msg_.ring_index_end[closest_scan_id + 1]; i++) {
        if (i == closest_idx)
            continue;
        if (!get_same && point_scan_id(cloud->points[i]) == closest_scan_id)
            continue;

        float d = square_distance(cloud->points[i], p);
        if (point_scan_id(cloud->points[i]) == closest_scan_id) {
            if (d < min_dis_same) {
                min_dis_same = d;
                min_idx_same = i;
            }
        } else {
            if (d < min_dis_adj) {
                min_dis_adj = d;
                min_idx_adj = i;
            }
        }
    }
    return {min_idx_same, min_idx_adj};
}

void FeatureAssociation::check_system_initialization() {
    corner_less_sharp_cloud_.swap(cloud_last_corner_);
    surf_less_flat_cloud_.swap(cloud_last_surf_);

    kd_last_corner_->setInputCloud(cloud_last_corner_);
    kd_last_surf_->setInputCloud(cloud_last_surf_);

    sensor_msgs::PointCloud2 laser_cloud_temp;

    pcl::toROSMsg(*cloud_last_corner_, laser_cloud_temp);
    laser_cloud_temp.header = cloud_header_;
    pub_last_corner_cloud_.publish(laser_cloud_temp);

    pcl::toROSMsg(*cloud_last_surf_, laser_cloud_temp);
    laser_cloud_temp.header = cloud_header_;
    pub_last_surf_cloud_.publish(laser_cloud_temp);

    reset_parameters();

    is_system_inited_ = true;
}

void FeatureAssociation::calculate_transformation() {
    if (cloud_last_corner_->points.size() < 10 || cloud_last_surf_->points.size() < 100)
        return;

    std::vector<int> closest_indices;
    std::vector<float> closest_square_distances;

    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

    ceres::Problem problem;
    problem.AddParameterBlock(pose_params_, 6);

    int correspondance_count = 0;

    spdlog::info("start surface ceres");
    for (const auto &sp : surf_flat_cloud_->points) {
      auto tp = transform_to_start(sp);
      kd_last_surf_->nearestKSearch(tp, 1, closest_indices, closest_square_distances);
      if (closest_square_distances[0] < nearestFeatureSearchSqDist) {
          auto indices = find_closest_in_same_adjacent_ring(closest_indices[0], tp, cloud_last_surf_, true); 

          if (indices[0] >= 0 && indices[1] >= 0) {
              const auto &points = cloud_last_surf_->points;

              problem.AddResidualBlock(new SurfCostFunction(sp, points[closest_indices[0]], points[indices[0]], points[indices[1]]), loss_function, pose_params_);
              correspondance_count++;
          }
      }
    }

    if (correspondance_count > 10) {
      solve_problem(problem);
    }

    spdlog::info("after surface ceres");
    correspondance_count = 0;
    for (const auto &cp : corner_sharp_cloud_->points) {
        auto tp = transform_to_start(cp);
        kd_last_corner_->nearestKSearch(tp, 1, closest_indices, closest_square_distances);
        if (closest_square_distances[0] < nearestFeatureSearchSqDist) {
            auto indices = find_closest_in_same_adjacent_ring(closest_indices[0], tp, cloud_last_corner_, false);

            if (indices[1] >= 0) {
                const auto &points = cloud_last_corner_->points;

                spdlog::info("before corner add residual block");
                problem.AddResidualBlock(new CornerCostFunction(cp, points[closest_indices[0]], points[indices[1]]), loss_function, pose_params_);
                correspondance_count++;
            }
        }
    }
    spdlog::info("before corner solve_problem");
    if (correspondance_count > 10) {
        solve_problem(problem);
    }
    spdlog::info("after corner ceres");

    q_last_curr_ = Eigen::AngleAxisd(pose_params_[5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pose_params_[4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(pose_params_[3], Eigen::Vector3d::UnitX());
    t_last_curr_ = Eigen::Vector3d{pose_params_[0], pose_params_[1], pose_params_[2]};
    t_w_curr_ += r_w_curr_ * t_last_curr_;
    r_w_curr_ = r_w_curr_ * q_last_curr_;

    static auto odom_frame_log = spdlog::basic_logger_st("odom_frame", "/home/pqf/my_lego/odom_frame_log.txt", true);
    odom_frame_log->info("tx, ty, tz: {}, {}, {}", t_last_curr_.x(), t_last_curr_.y(), t_last_curr_.z());
    odom_frame_log->info("rx, ry, rz: {}, {}, {}\n", q_last_curr_.x(), q_last_curr_.y(), q_last_curr_.z());
    odom_frame_log->flush();
}

void FeatureAssociation::publish_odometry() {
    laser_odom_.header.stamp = cloud_header_.stamp;
    laser_odom_.header.frame_id = "odom";
    laser_odom_.child_frame_id = "base_link";

    Eigen::Quaterniond q_w_curr(r_w_curr_);
    laser_odom_.pose.pose.orientation.x = q_w_curr.x();
    laser_odom_.pose.pose.orientation.y = q_w_curr.y();
    laser_odom_.pose.pose.orientation.z = q_w_curr.z();
    laser_odom_.pose.pose.orientation.w = q_w_curr.w();
    laser_odom_.pose.pose.position.x = t_w_curr_.x();
    laser_odom_.pose.pose.position.y = t_w_curr_.y();
    laser_odom_.pose.pose.position.z = t_w_curr_.z();
    pub_laser_odometry_.publish(laser_odom_);

    static auto odom_sum_log = spdlog::basic_logger_st("odom_sum", "/home/pqf/my_lego/odom_sum_log.txt", true);
    odom_sum_log->info("tx, ty, tz: {}, {}, {}", t_w_curr_.x(), t_w_curr_.y(), t_w_curr_.z());
    odom_sum_log->info("rx, ry, rz: {}, {}, {}\n", q_w_curr.x(), q_w_curr.y(), q_w_curr.z());
    odom_sum_log->flush();

    tf::Transform o2b;
    tf::poseMsgToTF(laser_odom_.pose.pose, o2b);
    tf_broad_.sendTransform(tf::StampedTransform(o2b, laser_odom_.header.stamp, "odom", "base_link"));
}

void FeatureAssociation::publish_cloud_last() {
    for (auto &p : corner_less_sharp_cloud_->points) {
        transform_to_end(p);
    }
    for (auto &p : surf_less_flat_cloud_->points) {
        transform_to_end(p);
    }

    corner_less_sharp_cloud_.swap(cloud_last_corner_);
    surf_less_flat_cloud_.swap(cloud_last_surf_);

    if (cloud_last_corner_->points.size()> 10 && cloud_last_surf_->points.size() > 100) {
        kd_last_corner_->setInputCloud(cloud_last_corner_);
        kd_last_surf_->setInputCloud(cloud_last_surf_);
    }

    sensor_msgs::PointCloud2 laser_cloud_temp;

    pcl::toROSMsg(*projected_outlier_cloud_, laser_cloud_temp);
    laser_cloud_temp.header = cloud_header_;
    pub_last_outlier_cloud_.publish(laser_cloud_temp);

    pcl::toROSMsg(*cloud_last_corner_, laser_cloud_temp);
    laser_cloud_temp.header = cloud_header_;
    pub_last_corner_cloud_.publish(laser_cloud_temp);

    pcl::toROSMsg(*cloud_last_surf_, laser_cloud_temp);
    laser_cloud_temp.header = cloud_header_;
    pub_last_surf_cloud_.publish(laser_cloud_temp);
}

void FeatureAssociation::run()
{
    if (has_get_cloud_ && has_get_cloud_msg_ && has_get_outlier_cloud_ &&
        std::abs(segment_cloud_info_time_ - segment_cloud_time_) < 0.05 &&
        std::abs(outlier_cloud_time_ - segment_cloud_time_) < 0.05) {
        has_get_cloud_ = false;
        has_get_cloud_msg_ = false;
        has_get_outlier_cloud_ = false;
    }else{
        return;
    }

    // adjust_distortion();

    calculate_smoothness();
    spdlog::info("after calculate smooth");

    mark_occluded_points();
    spdlog::info("after mark occluded points");

    extract_features();
    spdlog::info("after extract features");

 // for visualization
    publish_cloud();
    spdlog::info("after publish cloud for visualization");

    if (!is_system_inited_) {
        check_system_initialization();
        return;
    }

    calculate_transformation();
    spdlog::info("after calculate transformation");

    publish_odometry();
    spdlog::info("after publish odometry");

 // cloud to mapOptimization
    publish_cloud_last();
    spdlog::info("after publish cloud last");

    reset_parameters();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Feature Association Started.");

    FeatureAssociation fa;

    ros::Rate rate(100);
    while (ros::ok())
    {
        ros::spinOnce();
        fa.run();
        rate.sleep();
    }
    
    ros::spin();
    return 0;
}
