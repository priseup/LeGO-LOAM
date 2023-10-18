#include <cmath>
#include "utility.h"
#include "lego_math.h"

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
    pub_laser_odometry_ = nh_.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);

    cloud_last_corner_.reset(new pcl::PointCloud<Point>);
    cloud_last_surf_.reset(new pcl::PointCloud<Point>);
    coeff_sel_.reset(new pcl::PointCloud<Point>);

    kdtree_last_corner_.reset(new pcl::KdTreeFLANN<Point>);
    kdtree_last_surf_.reset(new pcl::KdTreeFLANN<Point>);

    laser_odometry_.header.frame_id = "camera_init";
    laser_odometry_.child_frame_id = "laser_odom";

    laser_odometry_trans_.frame_id_ = "camera_init";
    laser_odometry_trans_.child_frame_id_ = "laser_odom";
    
    projected_ground_segment_cloud_.reset(new pcl::PointCloud<Point>);
    projected_outlier_cloud_.reset(new pcl::PointCloud<Point>);

    corner_sharp_cloud_.reset(new pcl::PointCloud<Point>);
    corner_less_sharp_cloud_.reset(new pcl::PointCloud<Point>);
    surf_flat_cloud_.reset(new pcl::PointCloud<Point>);
    surf_less_flat_cloud_.reset(new pcl::PointCloud<Point>);

    init();
}

void FeatureAssociation::init()
{
    const int points_num = N_SCAN * Horizon_SCAN;

    cloud_smoothness_.resize(points_num);
    cloud_curvature_.resize(points_num);
    is_neibor_picked_.resize(points_num);
    cloud_label_.resize(points_num, FeatureAssociation::FeatureLabel::surf_less_flat);

    pointSearchCornerInd1.resize(points_num);
    pointSearchCornerInd2.resize(points_num);
    pointSearchSurfInd1.resize(points_num);
    pointSearchSurfInd2.resize(points_num);
    pointSearchSurfInd3.resize(points_num);

    voxel_grid_filter_.setLeafSize(0.2, 0.2, 0.2);

    mat_p_ = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
}

void FeatureAssociation::update_imu_rotation_start_sin_cos() {
    imu_cache_.roll_start_cos = std::cos(imu_cache_.roll_start);
    imu_cache_.pitch_start_cos = std::cos(imu_cache_.pitch_start);
    imu_cache_.yaw_start_cos = std::cos(imu_cache_.yaw_start);
    imu_cache_.roll_start_sin = std::sin(imu_cache_.roll_start);
    imu_cache_.pitch_start_sin = std::sin(imu_cache_.pitch_start);
    imu_cache_.yaw_start_sin = std::sin(imu_cache_.yaw_start);
}

void FeatureAssociation::shift_to_start_imu(const float &point_time)
{
    imu_cache_.drift_from_start_to_current_x = imu_cache_.shift_current_x - imu_cache_.shift_start_x - imu_cache_.vel_start_x * point_time;
    imu_cache_.drift_from_start_to_current_y = imu_cache_.shift_current_y - imu_cache_.shift_start_y - imu_cache_.vel_start_y * point_time;
    imu_cache_.drift_from_start_to_current_z = imu_cache_.shift_current_z - imu_cache_.shift_start_z - imu_cache_.vel_start_z * point_time;

    auto r0 = rotate_by_yxz_(imu_cache_.drift_from_start_to_current_x,
                            imu_cache_.drift_from_start_to_current_y,
                            imu_cache_.drift_from_start_to_current_z,
                            imu_cache_.pitch_start_cos,
                            -imu_cache_.pitch_start_sin,
                            imu_cache_.yaw_start_cos,
                            -imu_cache_.yaw_start_sin,
                            imu_cache_.roll_start_cos,
                            -imu_cache_.roll_start_sin);

    imu_cache_.drift_from_start_to_current_x = r0[0];
    imu_cache_.drift_from_start_to_current_y = r0[1];
    imu_cache_.drift_from_start_to_current_z = r0[2];
}

void FeatureAssociation::vel_to_start_imu()
{
    imu_cache_.vel_diff_from_start_to_current_x = imu_cache_.vel_current_x - imu_cache_.vel_start_x;
    imu_cache_.vel_diff_from_start_to_current_y = imu_cache_.vel_current_y - imu_cache_.vel_start_y;
    imu_cache_.vel_diff_from_start_to_current_z = imu_cache_.vel_current_z - imu_cache_.vel_start_z;

    auto r0 = rotate_by_yxz(imu_cache_.vel_diff_from_start_to_current_x,
                               imu_cache_.vel_diff_from_start_to_current_y,
                               imu_cache_.vel_diff_from_start_to_current_z,
                               imu_cache_.pitch_start_cos,
                               -imu_cache_.pitch_start_sin,
                               imu_cache_.yaw_start_cos,
                               -imu_cache_.yaw_start_sin,
                               imu_cache_.roll_start_cos,
                               -imu_cache_.roll_start_sin);

    imu_cache_.vel_diff_from_start_to_current_x = r0[0];
    imu_cache_.vel_diff_from_start_to_current_y = r0[1];
    imu_cache_.vel_diff_from_start_to_current_z = r0[2];
}

void FeatureAssociation::transform_to_start_imu(Point &p)
{
    auto r0 = rotate_by_zxy(p.x, p.y, p.z, imu_cache_.pitch_current, imu_cache_.yaw_current, imu_cache_.roll_current);
    auto r1 = rotate_by_yxz(r0[0], r0[1], r0[2],
                            imu_cache_.pitch_start_cos, -imu_cache_.pitch_start_sin,
                            imu_cache_.yaw_start_cos, -imu_cache_.yaw_start_sin,
                            imu_cache_.roll_start_cos, -imu_cache_.roll_start_sin);

    p->x = r1[0] + imu_cache_.drift_from_start_to_current_x;
    p->y = r1[1] + imu_cache_.drift_from_start_to_current_y;
    p->z = r1[2] + imu_cache_.drift_from_start_to_current_z;
}

void FeatureAssociation::accumulate_imu_shift_rotation()
{
    auto &imu_new = imu_cache_.imu_queue[imu_cache_.newest_idx];
    const auto &imu_last_new = imu_cache_.imu_queue[imu_cache_.dec(imu_cache_.newest_idx)];

    double time_diff = imu_new.time - imu_last_new.time;
    if (time_diff < scanPeriod) {
        auto r0 = rotate_by_zxy(imu_new.acc_x, imu_new.acc_y, imu_new.acc_z, imu_new.pitch, imu_new.yaw, imu_new.roll);

        imu_new.vel_x = imu_last_new.vel_x + r0[0] * time_diff;
        imu_new.vel_y = imu_last_new.vel_y + r0[1] * time_diff;
        imu_new.vel_z = imu_last_new.vel_z + r0[2] * time_diff;

        imu_new.shift_x = imu_last_new.shift_x + imu_last_new.vel_x * time_diff + shift_acc(r0[0], time_diff);
        imu_new.shift_y = imu_last_new.shift_y + imu_last_new.vel_y * time_diff + shift_acc(r0[1], time_diff);
        imu_new.shift_z = imu_last_new.shift_z + imu_last_new.vel_z * time_diff + shift_acc(r0[2], time_diff);

        imu_new.angular_x = imu_last_new.angular_x + imu_last_new.angular_vel_x * time_diff;
        imu_new.angular_y = imu_last_new.angular_y + imu_last_new.angular_vel_y * time_diff;
        imu_new.angular_z = imu_last_new.angular_z + imu_last_new.angular_vel_z * time_diff;
    }
}

void FeatureAssociation::imu_handler(const sensor_msgs::Imu::ConstPtr &imu)
{
    imu_cache_.newest_idx = imu_cache_.inc(imu_cache_.newest_idx);
    auto &imu_new = imu_cache_.imu_queue[imu_cache_.newest_idx];
    imu_new.time = imu->header.stamp.toSec();

    double roll, pitch, yaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(imu->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

    imu_new.roll = (float)roll;
    imu_new.pitch = (float)pitch;
    imu_new.yaw = (float)yaw;

    imu_new.acc_x = imu->linear_acceleration.y - std::sin(roll) * std::cos(pitch) * 9.81;
    imu_new.acc_y = imu->linear_acceleration.z - std::cos(roll) * std::cos(pitch) * 9.81;
    imu_new.acc_z = imu->linear_acceleration.x + std::sin(pitch) * 9.81;

    imu_new.angular_vel_x = imu->angular_velocity.x;
    imu_new.angular_vel_y = imu->angular_velocity.y;
    imu_new.angular_vel_z = imu->angular_velocity.z;

    accumulate_imu_shift_rotation();
}

void FeatureAssociation::laser_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& laser_cloud) {
    cloud_header_ = laser_cloud->header;

    laser_scan_time_ = cloud_header_.stamp.toSec();
    segment_cloud_time_ = laser_scan_time_;

    pcl::fromROSMsg(*laser_cloud, *projected_ground_segment_cloud_);

    has_get_cloud_ = true;
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

        float rx = projected_ground_segment_cloud_->points[i].x;
        float ry = projected_ground_segment_cloud_->points[i].y;
        float rz = projected_ground_segment_cloud_->points[i].z;
        point.x = ry;
        point.y = rz;
        point.z = rx;

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

        float point_time = (horizontal_angle - segmented_cloud_msg_.orientation_start) / segmented_cloud_msg_.orientation_diff * scanPeriod;
        point.intensity = int(point.intensity) + point_time;
        float laser_point_time = laser_scan_time_ + point_time;

        if (imu_cache_.newest_idx >= 0) {
            imu_cache_.after_laser_idx = imu_cache_.last_new_idx;
            while (laser_point_time < imu_cache_.imu_queue[imu_cache_.after_laser_idx].time) {
                if (imu_cache_.after_laser_idx != imu_cache_.newest_idx) {
                    break;
                }
                imu_cache_.after_laser_idx = imu_cache_.inc(imu_cache_.after_laser_idx);
            }

            const auto &imu_after_laser = imu_cache_.imu_queue[imu_cache_.after_laser_idx];
            if (laser_point_time > imu_after_laser.time) {
                // after_laser_idx == newest_idx, assign the newest imu to laser frame point state
                imu_cache_.roll_current = imu_after_laser.roll;
                imu_cache_.pitch_current = imu_after_laser.pitch;
                imu_cache_.yaw_current = imu_after_laser.yaw;

                imu_cache_.vel_current_x = imu_after_laser.vel_x;
                imu_cache_.vel_current_y = imu_after_laser.vel_y;
                imu_cache_.vel_current_z = imu_after_laser.vel_z;

                imu_cache_.shift_current_x = imu_after_laser.shift_x;
                imu_cache_.shift_current_y = imu_after_laser.shift_y;
                imu_cache_.shift_current_z = imu_after_laser.shift_z;

                if (i == 0) {
                    imu_cache_.angular_current_x = imu_after_laser.angular_x;
                    imu_cache_.angular_current_y = imu_after_laser.angular_y;
                    imu_cache_.angular_current_z = imu_after_laser.angular_z;
                }
            } else {
                int before_laser_idx = imu_cache_.dec(imu_cache_.after_laser_idx);
                const auto &imu_before_laser = imu_cache_.imu_queue[before_laser_idx];
                float ratio_from_start = (laser_point_time - imu_before_laser.time) 
                                        / (imu_after_laser.time - imu_before_laser.time);

                imu_cache_.roll_current = interpolation_by_linear(imu_before_laser.roll, imu_after_laser.roll, ratio_from_start);
                imu_cache_.pitch_current = interpolation_by_linear(imu_before_laser.pitch, imu_after_laser.pitch, ratio_from_start);

                float imu_before_yaw = imu_before_laser.yaw;
                if (imu_after_laser.yaw - imu_before_laser.yaw > M_PI) {
                    imu_before_yaw += 2 * M_PI;
                } else if (imu_after_laser.yaw - imu_before_laser.yaw < -M_PI) {
                    imu_before_yaw -= 2 * M_PI;
                }
                imu_cache_.yaw_current = interpolation_by_linear(imu_before_yaw, imu_after_laser.yaw, ratio_from_start);

                imu_cache_.vel_current_x = interpolation_by_linear(imu_before_laser.vel_x, imu_after_laser.vel_x, ratio_from_start);
                imu_cache_.vel_current_y = interpolation_by_linear(imu_before_laser.vel_y, imu_after_laser.vel_y, ratio_from_start);
                imu_cache_.vel_current_z = interpolation_by_linear(imu_before_laser.vel_z, imu_after_laser.vel_z, ratio_from_start);

                imu_cache_.shift_current_x = interpolation_by_linear(imu_before_laser.shift_x, imu_after_laser.shift_x, ratio_from_start);
                imu_cache_.shift_current_y = interpolation_by_linear(imu_before_laser.shift_y, imu_after_laser.shift_y, ratio_from_start);
                imu_cache_.shift_current_z = interpolation_by_linear(imu_before_laser.shift_z, imu_after_laser.shift_z, ratio_from_start);

                if (i == 0) {
                    imu_cache_.angular_current_x = interpolation_by_linear(imu_before_laser.angular_x, imu_after_laser.angular_x, ratio_from_start);
                    imu_cache_.angular_current_y = interpolation_by_linear(imu_before_laser.angular_y, imu_after_laser.angular_y, ratio_from_start);
                    imu_cache_.angular_current_z = interpolation_by_linear(imu_before_laser.angular_z, imu_after_laser.angular_z, ratio_from_start);
                }
            }

            if (i == 0) {
                imu_cache_.roll_start = imu_cache_.roll_current;
                imu_cache_.pitch_start = imu_cache_.pitch_current;
                imu_cache_.yaw_start = imu_cache_.yaw_current;

                imu_cache_.vel_start_x = imu_cache_.vel_current_x;
                imu_cache_.vel_start_y = imu_cache_.vel_current_y;
                imu_cache_.vel_start_z = imu_cache_.vel_current_z;

                imu_cache_.shift_start_x = imu_cache_.shift_current_x;
                imu_cache_.shift_start_y = imu_cache_.shift_current_y;
                imu_cache_.shift_start_z = imu_cache_.shift_current_z;

                // angular_current_x/y/z, is the first point in the current laser frame
                // last_angular_start_x/y/z, is the first point in the prev laser frame
                imu_cache_.angular_diff_from_prev_to_current_x = imu_cache_.angular_current_x - imu_cache_.last_angular_start_x;
                imu_cache_.angular_diff_from_prev_to_current_y = imu_cache_.angular_current_y - imu_cache_.last_angular_start_y;
                imu_cache_.angular_diff_from_prev_to_current_z = imu_cache_.angular_current_z - imu_cache_.last_angular_start_z;

                imu_cache_.last_angular_start_x = imu_cache_.angular_current_x;
                imu_cache_.last_angular_start_y = imu_cache_.angular_current_y;
                imu_cache_.last_angular_start_z = imu_cache_.angular_current_z;

                update_imu_rotation_start_sin_cos();
            } else {
                vel_to_start_imu();
                transform_to_start_imu(point);
            }
        }
    }
    imu_cache_.last_new_idx = imu_cache_.newest_idx;
}

void FeatureAssociation::calculate_smoothness()
{
    for (int i = 5; i < projected_ground_segment_cloud_->points.size() - 5; i++) {
        const auto &cloud_range = segmented_cloud_msg_.ground_segment_cloud_range; 
        float diff_range = cloud_range[i-5] + cloud_range[i-4]
                        + cloud_range[i-3] + cloud_range[i-2]
                        + cloud_range[i-1] - cloud_range[i] * 10
                        + cloud_range[i+1] + cloud_range[i+2]
                        + cloud_range[i+3] + cloud_range[i+4]
                        + cloud_range[i+5];            

        cloud_curvature_[i] = diff_range * diff_range;

        cloud_smoothness_[i].value = cloud_curvature_[i];
        cloud_smoothness_[i].idx = i;
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
        if (std::abs(cloud_column[idx + i] - cloud_column[idx + i - 1]) > 10)
            break;
        is_neibor_picked_[idx + i] = 1;
    }
    for (int i = -1; i >= -5; i--) {
        if (std::abs(cloud_column[idx + i] - cloud_column[idx + i + 1]) > 10)
            break;
        is_neibor_picked_[idx + i] = 1;
    }
}

void FeatureAssociation::extract_features()
{
    corner_sharp_cloud_->clear();
    corner_less_sharp_cloud_->clear();
    surf_flat_cloud_->clear();
    surf_less_flat_cloud_->clear();

    const auto &cloud = projected_ground_segment_cloud_->points;
    const auto &cloud_range = segmented_cloud_msg_.ground_segment_cloud_range;
    const auto &cloud_column = segmented_cloud_msg_.ground_segment_cloud_column;

    for (int i = 0; i < N_SCAN; i++) {
        for (int j = 0; j < 6; j++) {
            int sp = (segmented_cloud_msg_.ring_index_start[i] * (6 - j)  + segmented_cloud_msg_.ring_index_end[i] * j) / 6;
            int ep = (segmented_cloud_msg_.ring_index_start[i] * (5 - j)  + segmented_cloud_msg_.ring_index_end[i] * (j + 1)) / 6 - 1;
            if (sp >= ep)
                continue;
            std::sort(cloud_smoothness_.begin() + sp, cloud_smoothness_.begin() + ep + 1);

            for (int k = ep; ep >= sp; k--) {
                int idx = cloud_smoothness_[k].idx;

                if (is_neibor_picked_[idx] == 0
                    && !segmented_cloud_msg_.ground_segment_flag[idx]
                    && cloud_curvature_[idx] > edgeThreshold ) {
                    if (corner_sharp_cloud_->size() <= 2) {
                        cloud_label_[idx] = FeatureAssociation::FeatureLabel::corner_sharp;
                        corner_sharp_cloud_->push_back(cloud[idx]);
                        corner_less_sharp_cloud_->push_back(cloud[idx]);
                    } else if (corner_less_sharp_cloud_->size() <= 20) {
                        cloud_label_[idx] = FeatureAssociation::FeatureLabel::corner_less_sharp;
                        corner_less_sharp_cloud_->push_back(cloud[idx]);
                    } else {
                        break;
                    }

                    is_neibor_picked_[idx] = 1;
                    mark_neibor_is_picked(idx);
                }
            }

            for (int k = sp; k <= ep; k++) {
                int idx = cloud_smoothness_[k].idx;

                if (is_neibor_picked_[idx] == 0
                    && segmented_cloud_msg_.ground_segment_flag[idx]
                    && cloud_curvature_[idx] < surfThreshold) {
                    cloud_label_[idx] = FeatureAssociation::FeatureLabel::surf_flat;
                    surf_flat_cloud_->push_back(projected_ground_segment_cloud_->points[idx]);
                    surf_less_flat_cloud_->push_back(projected_ground_segment_cloud_->points[idx]);
                    if (surf_flat_cloud_->size() >= 4) {
                        break;
                    }

                    is_neibor_picked_[idx] = 1;
                    mark_neibor_is_picked(idx);
                }
            }

            for (int k = sp; k <= ep; k++) {
                if (cloud_label_[k] == FeatureAssociation::FeatureLabel::surf_less_flat) {
                    surf_less_flat_cloud_->push_back(projected_ground_segment_cloud_->points[k]);
                }
            }
        }
    }
    voxel_grid_filter_.setInputCloud(surf_less_flat_cloud_);
    voxel_grid_filter_.filter(*surf_less_flat_cloud_);
}

void FeatureAssociation::publish_cloud()
{
    sensor_msgs::PointCloud2 laser_cloud_msg;

    if (pub_corner_sharp_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*corner_sharp_cloud_, laser_cloud_msg);
        laser_cloud_msg.header.stamp = cloud_header_.stamp;
        laser_cloud_msg.header.frame_id = "camera";
        pub_corner_sharp_.publish(laser_cloud_msg);
    }

    if (pub_corner_less_sharp_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*corner_less_sharp_cloud_, laser_cloud_msg);
        laser_cloud_msg.header.stamp = cloud_header_.stamp;
        laser_cloud_msg.header.frame_id = "camera";
        pub_corner_less_sharp_.publish(laser_cloud_msg);
    }

    if (pub_surf_flat_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*surf_flat_cloud_, laser_cloud_msg);
        laser_cloud_msg.header.stamp = cloud_header_.stamp;
        laser_cloud_msg.header.frame_id = "camera";
        pub_surf_flat_.publish(laser_cloud_msg);
    }

    if (pub_surf_less_flat_.getNumSubscribers() != 0) {
        pcl::toROSMsg(*surf_less_flat_cloud_, laser_cloud_msg);
        laser_cloud_msg.header.stamp = cloud_header_.stamp;
        laser_cloud_msg.header.frame_id = "camera";
        pub_surf_less_flat_.publish(laser_cloud_msg);
    }
}

Point FeatureAssociation::transform_to_start(const Point &p)
{
    float s = 10 * (p.intensity - int(p.intensity));
    auto r = rotate_by_zxy(p.x - s * transform_from_prev_laser_frame_[3],
                           p.y - s * transform_from_prev_laser_frame_[4],
                           p.z - s * transform_from_prev_laser_frame_[5],
                           std::cos(s * transform_from_prev_laser_frame_[0]),
                           -std::sin(s * transform_from_prev_laser_frame_[0]),
                           std::cos(s * transform_from_prev_laser_frame_[1]),
                           -std::sin(s * transform_from_prev_laser_frame_[1]),
                           std::cos(s * transform_from_prev_laser_frame_[2]),
                           -std::sin(s * transform_from_prev_laser_frame_[2]));

    Point po;
    po.x = r[0];
    po.y = r[1];
    po.z = r[2];
    po.intensity = p.intensity;

    return po;
}

void FeaturpooooAssociation::transform_to_end(Point &p)
{
    Point po = transform_to_start(p);

    auto r0 = rotate_by_yxz(po.x, po.y, po.z,
                           std::cos(transform_from_prev_laser_frame_[0]),
                           std::sin(transform_from_prev_laser_frame_[0]),
                           std::cos(transform_from_prev_laser_frame_[1]),
                           std::sin(transform_from_prev_laser_frame_[1]),
                           std::cos(transform_from_prev_laser_frame_[2]),
                           std::sin(transform_from_prev_laser_frame_[2]));

    auto r1 = rotate_by_zxy(r0[0] + transform_from_prev_laser_frame_[3] - imu_cache_.drift_from_start_to_current_x,
                            r0[1] + transform_from_prev_laser_frame_[4] - imu_cache_.drift_from_start_to_current_y,
                            r0[2] + transform_from_prev_laser_frame_[5] - imu_cache_.drift_from_start_to_current_z,
                            imu_cache_.pitch_start_cos, imu_cache_.pitch_start_sin,
                            imu_cache_.yaw_start_cos, imu_cache_.yaw_start_sin,
                            imu_cache_.roll_start_cos, imu_cache_.roll_start_sin);
                            
    auto r2 = rotate_by_yxz(r1[0], r1[1], r1[2],
                            std::cos(imu_cache_.pitch_current), -std::sin(imu_cache_.pitch_current),
                            std::cos(imu_cache_.yaw_current), -std::sin(imu_cache_.yaw_current),
                            std::cos(imu_cache_.roll_current), -std::sin(imu_cache_.roll_current));

    p.x = r2[0];
    p.y = r2[1];
    p.z = r2[2];
    p.intensity = (int)p.intensity;
}

void FeatureAssociation::plugin_imu_rotation(const float &bcx, const float &bcy, const float &bcz,
                                            const float &blx, const float &bly, const float &blz, 
                                            const float &alx, const float &aly, const float &alz,
                                            float &acx, float &acy, float &acz)
{
    float sbcx = std::sin(bcx);
    float cbcx = std::cos(bcx);
    float sbcy = std::sin(bcy);
    float cbcy = std::cos(bcy);
    float sbcz = std::sin(bcz);
    float cbcz = std::cos(bcz);

    float sblx = std::sin(blx);
    float cblx = std::cos(blx);
    float sbly = std::sin(bly);
    float cbly = std::cos(bly);
    float sblz = std::sin(blz);
    float cblz = std::cos(blz);

    float salx = std::sin(alx);
    float calx = std::cos(alx);
    float saly = std::sin(aly);
    float caly = std::cos(aly);
    float salz = std::sin(alz);
    float calz = std::cos(alz);

    float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
                - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
    acx = -asin(srx);

    float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                    - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                    - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                    - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                    + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
    float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                    - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                    - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                    - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                    + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
    acy = std::atan2(srycrx / std::cos(acx), crycrx / std::cos(acx));
    
    float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
                    - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
                    - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
                    + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
                    - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
                    + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
                    + calx*cblx*salz*sblz);
    float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
                    - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
                    + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
                    + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
                    + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
                    - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
                    - calx*calz*cblx*sblz);
    acz = std::atan2(srzcrx / std::cos(acx), crzcrx / std::cos(acx));
}

void FeatureAssociation::accumulate_rotation(const float &cx, const float &cy, const float &cz,
                                            const float &lx, const float &ly, const float &lz, 
                                            float &ox, float &oy, float &oz)
{
    float srx = std::cos(lx)*std::cos(cx)*std::sin(ly)*std::sin(cz) - std::cos(cx)*std::cos(cz)*std::sin(lx) - std::cos(lx)*std::cos(ly)*std::sin(cx);
    ox = -asin(srx);

    float srycrx = std::sin(lx)*(std::cos(cy)*std::sin(cz) - std::cos(cz)*std::sin(cx)*std::sin(cy)) + std::cos(lx)*std::sin(ly)*(std::cos(cy)*std::cos(cz) 
                    + std::sin(cx)*std::sin(cy)*std::sin(cz)) + std::cos(lx)*std::cos(ly)*std::cos(cx)*std::sin(cy);
    float crycrx = std::cos(lx)*std::cos(ly)*std::cos(cx)*std::cos(cy) - std::cos(lx)*std::sin(ly)*(std::cos(cz)*std::sin(cy) 
                    - std::cos(cy)*std::sin(cx)*std::sin(cz)) - std::sin(lx)*(std::sin(cy)*std::sin(cz) + std::cos(cy)*std::cos(cz)*std::sin(cx));
    oy = std::atan2(srycrx / std::cos(ox), crycrx / std::cos(ox));

    float srzcrx = std::sin(cx)*(std::cos(lz)*std::sin(ly) - std::cos(ly)*std::sin(lx)*std::sin(lz)) + std::cos(cx)*std::sin(cz)*(std::cos(ly)*std::cos(lz) 
                    + std::sin(lx)*std::sin(ly)*std::sin(lz)) + std::cos(lx)*std::cos(cx)*std::cos(cz)*std::sin(lz);
    float crzcrx = std::cos(lx)*std::cos(lz)*std::cos(cx)*std::cos(cz) - std::cos(cx)*std::sin(cz)*(std::cos(ly)*std::sin(lz) 
                    - std::cos(lz)*std::sin(lx)*std::sin(ly)) - std::sin(cx)*(std::sin(ly)*std::sin(lz) + std::cos(ly)*std::cos(lz)*std::sin(lx));
    oz = std::atan2(srzcrx / std::cos(ox), crzcrx / std::cos(ox));
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

    for (int i = closest_idx + 1; i < cloud->points.size(); i++) {
        if (point_scan_id(cloud->points[i]) >= closest_scan_id + 2) {
            break; 
        }

        float d = square_distance(cloud->points[i], p);
        if (get_same && point_scan_id(cloud->points[i]) <= closest_scan_id) { // why not == closest_scan_id
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
        if (point_scan_id(cloud->points[i]) <= closest_scan_id - 2) {
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

    // for (int i = segmented_cloud_msg_.ring_index_start[closest_scan_id - 1]; i <= segmented_cloud_msg_.ring_index_end[closest_scan_id + 1]; i++) {
    //     if (i == closest_idx)
    //         continue;
        // if (!get_same && point_scan_id(cloud->points[i]) == closest_scan_id)
        //     continue;

    //     float d = square_distance(cloud->points[i], p);
    //     if (point_scan_id(cloud->points[i]) == closest_scan_id) { // why not == closest_scan_id
    //         if (d < min_dis_same) {
    //             min_dis_same = d;
    //             min_idx_same = i;
    //         }
    //     } else {
    //         if (d < min_dis_adj) {
    //             min_dis_adj = d;
    //             min_idx_adj = i;
    //         }
    //     }
    // }
}

void FeatureAssociation::find_corresponding_corner_features(int iterCount) {
    for (int i = 0; i < corner_sharp_cloud_->points.size(); i++) {
        if (iterCount % 5 == 0) {
            auto p = transform_to_start(corner_sharp_cloud_->points[i]);
            std::vector<int> closest_indices;
            std::vector<float> closest_square_distances;
            kdtree_last_corner_->nearestKSearch(p, 1, closest_indices, closest_square_distances);
            if (closest_square_distances[0] < nearestFeatureSearchSqDist) {
                auto ps = find_closest_in_same_adjacent_ring(closest_indices[0], p, cloud_last_corner_, false);

                pointSearchCornerInd1[i] = closest_indices[0];
                pointSearchCornerInd2[i] = ps[1];
            }
        }

        if (pointSearchCornerInd2[i] >= 0) {
            auto tripod1 = cloud_last_corner_->points[pointSearchCornerInd1[i]];
            auto tripod2 = cloud_last_corner_->points[pointSearchCornerInd2[i]];

            float x0 = p.x;
            float y0 = p.y;
            float z0 = p.z;
            float x1 = tripod1.x;
            float y1 = tripod1.y;
            float z1 = tripod1.z;
            float x2 = tripod2.x;
            float y2 = tripod2.y;
            float z2 = tripod2.z;

            float m11 = ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1));
            float m22 = ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1));
            float m33 = ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1));

            float a012 = std::sqrt(m11 * m11  + m22 * m22 + m33 * m33);
            float l12 = std::sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
            float la =  ((y1 - y2)*m11 + (z1 - z2)*m22) / a012 / l12;
            float lb = -((x1 - x2)*m11 - (z1 - z2)*m33) / a012 / l12;
            float lc = -((x1 - x2)*m22 + (y1 - y2)*m33) / a012 / l12;
            float ld2 = a012 / l12;

            float s = 1;
            if (iterCount >= 5) {
                s = 1 - 1.8 * fabs(ld2);
            }

            if (s > 0.1 && ld2 != 0) {
                Point coeff;
                coeff.x = s * la;
                coeff.y = s * lb;
                coeff.z = s * sc;
                coeff.intensity = s * ld2;
                coeff_sel_->push_back(coeff);
                cloud_ori_indices_.push_back(i);
            }
        }
    }
}

void FeatureAssociation::find_corresponding_surf_features(int iterCount) {
    for (int i = 0; i < surf_flat_cloud_->points.size(); i++) {
        if (iterCount % 5 == 0) {
            auto p = transform_to_start(surf_flat_cloud_->points[i]);
            std::vector<int> closest_indices;
            std::vector<float> closest_square_distances;
            kdtree_last_surf_->nearestKSearch(p, 1, closest_indices, closest_square_distances);

            if (closest_square_distances[0] < nearestFeatureSearchSqDist) {
                auto ps = find_closest_in_same_adjacent_ring(closest_indices[0], p, cloud_last_surf_, true); 

                pointSearchSurfInd1[i] = closest_indices[0];
                pointSearchSurfInd2[i] = ps[0];
                pointSearchSurfInd3[i] = ps[1];
            }
        }

        if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {
            auto tripod1 = cloud_last_surf_->points[pointSearchSurfInd1[i]];
            auto tripod2 = cloud_last_surf_->points[pointSearchSurfInd2[i]];
            auto tripod3 = cloud_last_surf_->points[pointSearchSurfInd3[i]];

            float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                        - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
            float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                        - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
            float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                        - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
            float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

            float ps = sqrt(pa * pa + pb * pb + pc * pc);

            pa /= ps;
            pb /= ps;
            pc /= ps;
            pd /= ps;

            float pd2 = pa * p.x + pb * p.y + pc * p.z + pd;

            float s = 1;
            if (iterCount >= 5) {
                s = 1 - 1.8 * fabs(pd2) / sqrt(laser_range(p));
            }

            if (s > 0.1 && pd2 != 0) {
                Point coeff;
                coeff.x = s * pa;
                coeff.y = s * pb;
                coeff.z = s * pc;
                coeff.intensity = s * pd2;
                coeff_sel_->push_back(coeff);
                cloud_ori_indices_.push_back(i);
            }
        }
    }
}

bool FeatureAssociation::calculate_suf_transformation(int iterCount) {
    int pointSelNum = cloud_ori_indices_.size();

    cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

    float srx = std::sin(transform_from_prev_laser_frame_[0]);
    float crx = std::cos(transform_from_prev_laser_frame_[0]);
    float sry = std::sin(transform_from_prev_laser_frame_[1]);
    float cry = std::cos(transform_from_prev_laser_frame_[1]);
    float srz = std::sin(transform_from_prev_laser_frame_[2]);
    float crz = std::cos(transform_from_prev_laser_frame_[2]);
    float tx = transform_from_prev_laser_frame_[3];
    float ty = transform_from_prev_laser_frame_[4];
    float tz = transform_from_prev_laser_frame_[5];

    float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
    float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
    float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

    float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz;
    float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry;

    float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
    float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

    for (int i = 0; i < pointSelNum; i++) {
        const auto &p = surf_flat_cloud_->points[cloud_ori_indices_[i]];
        const auto &coeff = coeff_sel_->points[i];

        float arx = (-a1*p.x + a2*p.y + a3*p.z + a4) * coeff.x
                    + (a5*p.x - a6*p.y + crx*p.z + a7) * coeff.y
                    + (a8*p.x - a9*p.y - a10*p.z + a11) * coeff.z;

        float arz = (c1*p.x + c2*p.y + c3) * .x
                    + (c4*p.x - c5*p.y + c6) * coeff.y
                    + (c7*p.x + c8*p.y + c9) * coeff.z;

        float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

        float d2 = coeff.intensity;

        matA.at<float>(i, 0) = arx;
        matA.at<float>(i, 1) = arz;
        matA.at<float>(i, 2) = aty;
        matB.at<float>(i, 0) = -0.05 * d2;
    }

    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (iterCount == 0) {
        cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

        cv::eigen(matAtA, matE, matV);
        matV.copyTo(matV2);

        is_degenerate_ = false;
        float eignThre[3] = {10, 10, 10};
        for (int i = 2; i >= 0; i--) {
            if (matE.at<float>(0, i) < eignThre[i]) {
                for (int j = 0; j < 3; j++) {
                    matV2.at<float>(i, j) = 0;
                }
                is_degenerate_ = true;
            } else {
                break;
            }
        }
        mat_p_ = matV.inv() * matV2;
    }

    if (is_degenerate_) {
        cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
        matX.copyTo(matX2);
        matX = mat_p_ * matX2;
    }

    transform_from_prev_laser_frame_[0] += matX.at<float>(0, 0);
    transform_from_prev_laser_frame_[2] += matX.at<float>(1, 0);
    transform_from_prev_laser_frame_[4] += matX.at<float>(2, 0);

    for(auto &e : transform_from_prev_laser_frame_) {
        if(std::isnan(e))
            e = 0.f;
    }

    float deltaR = sqrt(
                        pow(rad2deg(matX.at<float>(0, 0)), 2) +
                        pow(rad2deg(matX.at<float>(1, 0)), 2));
    float deltaT = sqrt(
                        pow(matX.at<float>(2, 0) * 100, 2));

    if (deltaR < 0.1 && deltaT < 0.1) {
        return false;
    }
    return true;
}

bool FeatureAssociation::calculate_corner_transformation(int iterCount) {
    int pointSelNum = cloud_ori_indices_.size();

    cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

    float srx = std::sin(transform_from_prev_laser_frame_[0]);
    float crx = std::cos(transform_from_prev_laser_frame_[0]);
    float sry = std::sin(transform_from_prev_laser_frame_[1]);
    float cry = std::cos(transform_from_prev_laser_frame_[1]);
    float srz = std::sin(transform_from_prev_laser_frame_[2]);
    float crz = std::cos(transform_from_prev_laser_frame_[2]);
    float tx = transform_from_prev_laser_frame_[3];
    float ty = transform_from_prev_laser_frame_[4];
    float tz = transform_from_prev_laser_frame_[5];

    float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
    float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

    float c5 = crx*srz;

    for (int i = 0; i < pointSelNum; i++) {
        const auto &p = corner_sharp_cloud_->points[cloud_ori_indices_[i]];
        const auto &coeff = coeff_sel_->points[i];

        float ary = (b1*p.x + b2*p.y - b3*p.z + b4) * coeff.x
                    + (b5*p.x + b6*p.y - b7*p.z + b8) * coeff.z;

        float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

        float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

        float d2 = coeff.intensity;

        matA.at<float>(i, 0) = ary;
        matA.at<float>(i, 1) = atx;
        matA.at<float>(i, 2) = atz;
        matB.at<float>(i, 0) = -0.05 * d2;
    }

    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (iterCount == 0) {
        cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

        cv::eigen(matAtA, matE, matV);
        matV.copyTo(matV2);

        is_degenerate_ = false;
        float eignThre[3] = {10, 10, 10};
        for (int i = 2; i >= 0; i--) {
            if (matE.at<float>(0, i) < eignThre[i]) {
                for (int j = 0; j < 3; j++) {
                    matV2.at<float>(i, j) = 0;
                }
                is_degenerate_ = true;
            } else {
                break;
            }
        }
        mat_p_ = matV.inv() * matV2;
    }

    if (is_degenerate_) {
        cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
        matX.copyTo(matX2);
        matX = mat_p_ * matX2;
    }

    transform_from_prev_laser_frame_[1] += matX.at<float>(0, 0);
    transform_from_prev_laser_frame_[3] += matX.at<float>(1, 0);
    transform_from_prev_laser_frame_[5] += matX.at<float>(2, 0);

    for(auto &e : transform_from_prev_laser_frame_) {
        if(std::isnan(e))
            e = 0.f;
    }

    float deltaR = sqrt(
                        pow(rad2deg(matX.at<float>(0, 0)), 2));
    float deltaT = sqrt(
                        pow(matX.at<float>(1, 0) * 100, 2) +
                        pow(matX.at<float>(2, 0) * 100, 2));

    if (deltaR < 0.1 && deltaT < 0.1) {
        return false;
    }
    return true;
}

void FeatureAssociation::check_system_initialization() {
    corner_less_sharp_cloud_.swap(cloud_last_corner_);
    surf_less_flat_cloud_.swap(cloud_last_surf_);

    kdtree_last_corner_->setInputCloud(cloud_last_corner_);
    kdtree_last_surf_->setInputCloud(cloud_last_surf_);

    sensor_msgs::PointCloud2 laser_cloud_temp;

    pcl::toROSMsg(*cloud_last_corner_, laser_cloud_temp);
    laser_cloud_temp.header.stamp = cloud_header_.stamp;
    laser_cloud_temp.header.frame_id = "camera";
    pub_last_corner_cloud_.publish(laser_cloud_temp);

    pcl::toROSMsg(*cloud_last_surf_, laser_cloud_temp);
    laser_cloud_temp.header.stamp = cloud_header_.stamp;
    laser_cloud_temp.header.frame_id = "camera";
    pub_last_surf_cloud_.publish(laser_cloud_temp);

    transform_from_first_laser_frame_[0] += imu_cache_.pitch_start;
    transform_from_first_laser_frame_[2] += imu_cache_.roll_start;

    is_system_inited_ = true;
}

void FeatureAssociation::update_initial_guess() {
    if (imu_cache_.angular_diff_from_prev_to_current_x != 0 || imu_cache_.angular_diff_from_prev_to_current_y != 0 || imu_cache_.angular_diff_from_prev_to_current_z != 0) {
        transform_from_prev_laser_frame_[0] = -imu_cache_.angular_diff_from_prev_to_current_y;  // why -angular_diff
        transform_from_prev_laser_frame_[1] = -imu_cache_.angular_diff_from_prev_to_current_z;
        transform_from_prev_laser_frame_[2] = -imu_cache_.angular_diff_from_prev_to_current_x;
    }
    
    if (imu_cache_.vel_diff_from_start_to_current_x != 0 || imu_cache_.vel_diff_from_start_to_current_y != 0 || imu_cache_.vel_diff_from_start_to_current_z != 0) {
        transform_from_prev_laser_frame_[3] -= imu_cache_.vel_diff_from_start_to_current_x * scanPeriod;    // why -= 
        transform_from_prev_laser_frame_[4] -= imu_cache_.vel_diff_from_start_to_current_y * scanPeriod;
        transform_from_prev_laser_frame_[5] -= imu_cache_.vel_diff_from_start_to_current_z * scanPeriod;
    }
}

void FeatureAssociation::update_transformation() {
    if (cloud_last_corner_->points.size() < 10 || cloud_last_surf_->points.size() < 100)
        return;

    for (int i = 0; i < 25; i++) {
        cloud_ori_indices_.clear();
        coeff_sel_->clear();

        find_corresponding_surf_features(i);
        if (cloud_ori_indices_.size() < 10)
            continue;
        if (!calculate_suf_transformation(i))
            break;
    }

    for (int i = 0; i < 25; i++) {
        cloud_ori_indices_.clear();
        coeff_sel_->clear();

        find_corresponding_corner_features(i);
        if (cloud_ori_indices_.size() < 10)
            continue;
        if (!calculate_corner_transformation(i))
            break;
    }
}

void FeatureAssociation::integrate_transformation() {
    float rx, ry, rz;
    accumulate_rotation(transform_from_first_laser_frame_[0], transform_from_first_laser_frame_[1], transform_from_first_laser_frame_[2], 
                        -transform_from_prev_laser_frame_[0], -transform_from_prev_laser_frame_[1], -transform_from_prev_laser_frame_[2], rx, ry, rz);

    auto r = rotate_by_zxy(transform_from_prev_laser_frame_[3] - imu_cache_.drift_from_start_to_current_x,
                           transform_from_prev_laser_frame_[4] - imu_cache_.drift_from_start_to_current_y,
                           transform_from_prev_laser_frame_[5] - imu_cache_.drift_from_start_to_current_z,
                           std::cos(rx), std::sin(rx),
                           std::cos(ry), std::sin(ry),
                           std::cos(rz), std::sin(rz));

    plugin_imu_rotation(rx, ry, rz, imu_cache_.pitch_start, imu_cache_.yaw_start, imu_cache_.roll_start, 
                        imu_cache_.pitch_current,
                        imu_cache_.yaw_current,
                        imu_cache_.roll_current,
                        rx, ry, rz);

    transform_from_first_laser_frame_[0] = rx;
    transform_from_first_laser_frame_[1] = ry;
    transform_from_first_laser_frame_[2] = rz;
    transform_from_first_laser_frame_[3] -= r[0];
    transform_from_first_laser_frame_[4] -= r[1];
    transform_from_first_laser_frame_[5] -= r[2];
}

void FeatureAssociation::publish_odometry() {
    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transform_from_first_laser_frame_[2], -transform_from_first_laser_frame_[0], -transform_from_first_laser_frame_[1]);

    laser_odometry_.header.stamp = cloud_header_.stamp;
    laser_odometry_.pose.pose.orientation.x = -geoQuat.y;
    laser_odometry_.pose.pose.orientation.y = -geoQuat.z;
    laser_odometry_.pose.pose.orientation.z = geoQuat.x;
    laser_odometry_.pose.pose.orientation.w = geoQuat.w;
    laser_odometry_.pose.pose.position.x = transform_from_first_laser_frame_[3];
    laser_odometry_.pose.pose.position.y = transform_from_first_laser_frame_[4];
    laser_odometry_.pose.pose.position.z = transform_from_first_laser_frame_[5];
    pub_laser_odometry_.publish(laser_odometry_);

    laser_odometry_trans_.stamp_ = cloud_header_.stamp;
    laser_odometry_trans_.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    laser_odometry_trans_.setOrigin(tf::Vector3(transform_from_first_laser_frame_[3], transform_from_first_laser_frame_[4], transform_from_first_laser_frame_[5]));
    tf_broadcaster_.sendTransform(laser_odometry_trans_);
}

void FeatureAssociation::adjust_outlier_cloud() {
    for (auto &p : projected_outlier_cloud_->points)
    {
        float rx = p.x;
        float ry = p.y;
        float rz = p.z;
        point.x = ry;
        point.y = rz;
        point.z = rx;
    }
}

void FeatureAssociation::publish_cloud_last() {
    static int frame_count = 0;
    // update_imu_rotation_start_sin_cos();
    for (auto &p : corner_less_sharp_cloud_->points) {
        transform_to_end(p);
    }
    for (auto &p : surf_less_flat_cloud_->points) {
        transform_to_end(p);
    }

    corner_less_sharp_cloud_.swap(cloud_last_corner_);
    surf_less_flat_cloud_.swap(cloud_last_surf_);

    if (cloud_last_corner_->points.size()> 10 && cloud_last_surf_->points.size() > 100) {
        kdtree_last_corner_->setInputCloud(cloud_last_corner_);
        kdtree_last_surf_->setInputCloud(cloud_last_surf_);
    }

    if (frame_count % 2) {
        adjust_outlier_cloud();

        sensor_msgs::PointCloud2 laser_cloud_temp;

        pcl::toROSMsg(*projected_outlier_cloud_, laser_cloud_temp);
        laser_cloud_temp.header.stamp = cloud_header_.stamp;
        laser_cloud_temp.header.frame_id = "camera";
        pub_last_outlier_cloud_.publish(laser_cloud_temp);

        pcl::toROSMsg(*cloud_last_corner_, laser_cloud_temp);
        laser_cloud_temp.header.stamp = cloud_header_.stamp;
        laser_cloud_temp.header.frame_id = "camera";
        pub_last_corner_cloud_.publish(laser_cloud_temp);

        pcl::toROSMsg(*cloud_last_surf_, laser_cloud_temp);
        laser_cloud_temp.header.stamp = cloud_header_.stamp;
        laser_cloud_temp.header.frame_id = "camera";
        pub_last_surf_cloud_.publish(laser_cloud_temp);
    }
    if (frame_count++ >= 100)
        frame_count = 0;
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

    adjust_distortion();

    calculate_smoothness();

    mark_occluded_points();

    extract_features();

    publish_cloud(); // for visualization

    if (!is_system_inited_) {
        check_system_initialization();
        return;
    }

    update_initial_guess();

    update_transformation();

    integrate_transformation();

    publish_odometry();

    publish_cloud_last(); // cloud to mapOptimization
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Feature Association Started.");

    FeatureAssociation fa;

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();
        fa.run();
        rate.sleep();
    }
    
    ros::spin();
    return 0;
}
