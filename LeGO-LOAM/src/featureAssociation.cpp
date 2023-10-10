#include <cmath>
#include "utility.h"
#include "lego_math.h"

FeatureAssociation::FeatureAssociation(): nh_("~") {
    sub_laser_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &FeatureAssociation::laser_cloud_handler, this);
    sub_laser_cloud_info_ = nh_.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &FeatureAssociation::laser_cloud_msg_handler, this);
    sub_outlier_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &FeatureAssociation::outlier_cloud_handler, this);
    sub_imu_ = nh_.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureAssociation::imu_handler, this);

    pub_corner_sharp_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
    pub_corner_less_sharp_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
    pub_surf_flat_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
    pub_surf_less_flat_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

    pub_last_corner_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
    pub_last_surf_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
    pub_last_outlier_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
    pub_laser_odometry_ = nh_.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);

    cloud_last_corner_.reset(new pcl::PointCloud<Point>);
    cloud_last_surf_.reset(new pcl::PointCloud<Point>);
    cloud_ori_.reset(new pcl::PointCloud<Point>);
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

    surf_less_flat_scan_.reset(new pcl::PointCloud<Point>);
    surf_less_flat_scan_ds_.reset(new pcl::PointCloud<Point>);

    init();
}

void init()
{
    const int points_num = N_SCAN * Horizon_SCAN;

    cloud_curvature_ = new float[points_num];
    cloudNeighborPicked = new int[points_num];
    cloud_label_ = new int[points_num];

    pointSearchCornerInd1 = new float[points_num];
    pointSearchCornerInd2 = new float[points_num];

    pointSearchSurfInd1 = new float[points_num];
    pointSearchSurfInd2 = new float[points_num];
    pointSearchSurfInd3 = new float[points_num];

    cloud_smoothness_.resize(points_num);

    voxel_grid_filter_.setLeafSize(0.2, 0.2, 0.2);

    for (int i = 0; i < 6; ++i) {
        transformCur[i] = 0;
        transform_sum_[i] = 0;
    }

    mat_p_ = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
}

void updateImuRollPitchYawStartSinCos() {
    imu_cache.roll_start_cos = std::cos(imu_cache.roll_start);
    imu_cache.pitch_start_cos = std::cos(imu_cache.pitch_start);
    imu_cache.yaw_start_cos = std::cos(imu_cache.yaw_start);
    imu_cache.roll_start_sin = std::sin(imu_cache.roll_start);
    imu_cache.pitch_start_sin = std::sin(imu_cache.pitch_start);
    imu_cache.yaw_start_sin = std::sin(imu_cache.yaw_start);
}

void shift_to_start_imu(float point_time)
{
    imu_cache.drift_from_start_to_current_x = imu_cache.shift_current_x - imu_cache.shift_start_x - imu_cache.vel_start_x * point_time;
    imu_cache.drift_from_start_to_current_y = imu_cache.shift_current_y - imu_cache.shift_start_y - imu_cache.vel_start_y * point_time;
    imu_cache.drift_from_start_to_current_z = imu_cache.shift_current_z - imu_cache.shift_start_z - imu_cache.vel_start_z * point_time;

    auto r0 = rotate_by_yxz_(imu_cache.drift_from_start_to_current_x,
                            imu_cache.drift_from_start_to_current_y,
                            imu_cache.drift_from_start_to_current_z,
                            imu_cache.roll_start_cos,
                            -imu_cache.roll_start_sin,
                            imu_cache.pitch_start_cos,
                            -imu_cache.pitch_start_sin,
                            imu_cache.yaw_start_cos,
                            -imu_cache.yaw_start_sin);

    imu_cache.drift_from_start_to_current_x = r0[0];
    imu_cache.drift_from_start_to_current_y = r0[1];
    imu_cache.drift_from_start_to_current_z = r0[2];
}

void vel_to_start_imu()
{
    imu_cache.vel_diff_from_start_to_current_x = imu_cache.vel_current_x - imu_cache.vel_start_x;
    imu_cache.vel_diff_from_start_to_current_y = imu_cache.vel_current_y - imu_cache.vel_start_y;
    imu_cache.vel_diff_from_start_to_current_z = imu_cache.vel_current_z - imu_cache.vel_start_z;

    auto r0 = rotate_by_yxz(imu_cache.vel_diff_from_start_to_current_x,
                               imu_cache.vel_diff_from_start_to_current_y,
                               imu_cache.vel_diff_from_start_to_current_z,
                               imu_cache.roll_start_cos,
                               -imu_cache.roll_start_sin,
                               imu_cache.pitch_start_cos,
                               -imu_cache.pitch_start_sin,
                               imu_cache.yaw_start_cos,
                               -imu_cache.yaw_start_sin);

    imu_cache.vel_diff_from_start_to_current_x = r0[0];
    imu_cache.vel_diff_from_start_to_current_y = r0[1];
    imu_cache.vel_diff_from_start_to_current_z = r0[2];
}

void transform_to_start_imu(Point &p)
{
    auto r0 = rotate_by_zxy(p.x, p.y, p.z, imu_cache.roll_current, imu_cache.pitch_current, imu_cache.yaw_current);
    auto r1 = rotate_by_yxz(r0[0], r0[1], r0[2],
                            imu_cache.roll_start_cos, -imu_cache.roll_start_sin,
                            imu_cache.pitch_start_cos, -imu_cache.pitch_start_sin,
                            imu_cache.yaw_start_cos, -imu_cache.yaw_start_sin);

    p->x = r1[0] + imu_cache.drift_from_start_to_current_x_;
    p->y = r1[1] + imu_cache.drift_from_start_to_current_y_;
    p->z = r1[2] + imu_cache.drift_from_start_to_current_z_;
}

void accumulate_imu_shift_rotation()
{
    auto &imu_new = imu_cache.imu_queue[imu_cache.newest_idx];
    const auto &imu_last_new = imu_cache.imu_queue[imu_cache.idx_decrement(imu_cache.newest_idx)];

    double time_diff = imu_new.time - imu_last_new.time;
    if (time_diff < scanPeriod) {
        auto r0 = rotate_by_zxy(imu_new.acc_x, imu_new.acc_y, imu_new.acc_z, imu_new.roll, imu_new.pitch, imu_new.yaw);

        imu_new.vel_x = imu_last_new.vel_x + r0[0] * time_diff;
        imu_new.vel_y = imu_last_new.vel_y + r0[1] * time_diff;
        imu_new.vel_z = imu_last_new.vel_z + r0[2] * time_diff;

        imu_new.shift_x = imu_last_new.shift_x + imu_last_new.vel_x * time_diff + r0[0] * time_diff * time_diff / 2;
        imu_new.shift_y = imu_last_new.shift_y + imu_last_new.vel_y * time_diff + r0[1] * time_diff * time_diff / 2;
        imu_new.shift_z = imu_last_new.shift_z + imu_last_new.vel_z * time_diff + r0[2] * time_diff * time_diff / 2;

        imu_new.angular_rotation_x = imu_last_new.angular_rotation_x + imu_last_new.angular_vel_x * time_diff;
        imu_new.angular_rotation_y = imu_last_new.angular_rotation_y + imu_last_new.angular_vel_y * time_diff;
        imu_new.angular_rotation_z = imu_last_new.angular_rotation_z + imu_last_new.angular_vel_z * time_diff;
    }
}

void imu_handler(const sensor_msgs::Imu::ConstPtr &imu)
{
    imu_cache.newest_idx = imu_cache.idx_increment(imu_cache.newest_idx);
    auto &imu_new = imu_cache.imu_queue[imu_cache.newest_idx];
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

void laser_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& laser_cloud) {
    cloud_header_ = laser_cloud->header;

    laser_scan_time_ = cloud_header_.stamp.toSec();
    segment_cloud_time_ = laser_scan_time_;

    projected_ground_segment_cloud_->clear();
    pcl::fromROSMsg(*laser_cloud, *projected_ground_segment_cloud_);

    has_get_cloud_ = true;
}

void outlier_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& msgIn) {
    outlier_cloud_time_ = msgIn->header.stamp.toSec();

    projected_outlier_cloud_->clear();
    pcl::fromROSMsg(*msgIn, *projected_outlier_cloud_);

    has_get_outlier_cloud_ = true;
}

void laser_cloud_msg_handler(const cloud_msgs::cloud_infoConstPtr& msgIn) {
    segment_cloud_info_time_ = msgIn->header.stamp.toSec();
    segmented_cloud_msg_ = *msgIn;

    has_get_cloud_msg_ = true;
}

void adjust_distortion() {
    bool is_half_pass = false;

    for (int i = 0; i < projected_ground_segment_cloud_->points.size(); i++) {
        auto &point = projected_ground_segment_cloud_->points[i];
        float horizontal_angle = -std::atan2(point.y, point.x);

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

        if (imu_cache.newest_idx >= 0) {
            imu_cache.after_laser_idx = imu_cache.newest_idxIteration;
            while (imu_cache.after_laser_idx != imu_cache.newest_idx) {
                if (laser_point_time < imu_cache.imu_queue[imu_cache.after_laser_idx].time) {
                    break;
                }
                imu_cache.after_laser_idx = imu.idx_increment(imu.after_laser_idx);
            }

            const auto &imu_after_laser = imu_cache.imu_queue[imu_cache.after_laser_idx];
            if (laser_point_time > imu_after_laser.time) {
                // assign the newest imu to current state
                imu_cache.roll_current = imu_after_laser.roll;
                imu_cache.pitch_current = imu_after_laser.pitch;
                imu_cache.yaw_current = imu_after_laser.yaw;

                imu_cache.vel_current_x = imu_after_laser.vel_x;
                imu_cache.vel_current_y = imu_after_laser.vel_y;
                imu_cache.vel_current_z = imu_after_laser.vel_z;

                imu_cache.shift_current_x = imu_after_laser.shift_x;
                imu_cache.shift_current_y = imu_after_laser.shift_y;
                imu_cache.shift_current_z = imu_after_laser.shift_z;
            } else {
                int before_laser_idx = imu_cache.idx_decrement(imu_cache.after_laser_idx);
                const auto &imu_before_laser = imu_cache.imu_queue[before_laser_idx];
                float ratio_from_start = (laser_point_time - imu_before_laser.time) 
                                        / (imu_after_laser.time - imu_before_laser.time);

                imu_cache.roll_current = interpolation_by_linear(imu_before_laser.roll, imu_after_laser.roll, ratio_from_start);
                imu_cache.pitch_current = interpolation_by_linear(imu_before_laser.pitch, imu_after_laser.pitch, ratio_from_start);
                if (imu_after_laser.yaw - imu_before_laser.yaw > M_PI) {
                    imu_cache.yaw_current = interpolation_by_linear(imu_before_laser.yaw + 2 * M_PI, imu_after_laser.yaw, ratio_from_start);
                } else if (imu_after_laser.yaw - imu_before_laser.yaw < -M_PI) {
                    imu_cache.yaw_current = interpolation_by_linear(imu_before_laser.yaw - 2 * M_PI, imu_after_laser.yaw, ratio_from_start);
                } else {
                    imu_cache.yaw_current = interpolation_by_linear(imu_before_laser.yaw, imu_after_laser.yaw, ratio_from_start);
                }

                imu_cache.vel_current_x = interpolation_by_linear(imu_before_laser.vel_x, imu_after_laser.vel_x, ratio_from_start);
                imu_cache.vel_current_y = interpolation_by_linear(imu_before_laser.vel_y, imu_after_laser.vel_y, ratio_from_start);
                imu_cache.vel_current_z = interpolation_by_linear(imu_before_laser.vel_z, imu_after_laser.vel_z, ratio_from_start);

                imu_cache.shift_current_x = interpolation_by_linear(imu_before_laser.shift_x, imu_after_laser.shift_x, ratio_from_start);
                imu_cache.shift_current_y = interpolation_by_linear(imu_before_laser.shift_y, imu_after_laser.shift_y, ratio_from_start);
                imu_cache.shift_current_z = interpolation_by_linear(imu_before_laser.shift_z, imu_after_laser.shift_z, ratio_from_start);
            }

            if (i == 0) {
                imu_cache.roll_start = imu_cache.roll_current;
                imu_cache.pitch_start = imu_cache.pitch_current;
                imu_cache.yaw_start = imu_cache.yaw_current;

                imu_cache.vel_start_x = imu_cache.vel_current_x;
                imu_cache.vel_start_y = imu_cache.vel_current_y;
                imu_cache.vel_start_z = imu_cache.vel_current_z;

                imu_cache.shift_start_x = imu_cache.shift_current_x;
                imu_cache.shift_start_y = imu_cache.shift_current_y;
                imu_cache.shift_start_z = imu_cache.shift_current_z;

                if (laser_point_time > imu_after_laser.time) {
                    imu_cache.angular_rotation_current_x = imu_after_laser.angular_rotation_x;
                    imu_cache.angular_rotation_current_y = imu_after_laser.angular_rotation_y;
                    imu_cache.angular_rotation_current_z = imu_after_laser.angular_rotation_z;
                }else{
                    int before_laser_idx = imu_cache.idx_decrement(imu_cache.after_laser_idx);
                    const auto &imu_before_laser = imu_cache.imu_queue[before_laser_idx];
                    float ratio_from_start = (laser_point_time - imu_before_laser.time) 
                                            / (imu_after_laser.time - imu_before_laser.time);

                    imu_cache.angular_rotation_current_x = interpolation_by_linear(imu_before_laser.angular_rotation_x, imu_after_laser.angular_rotation_x, ratio_from_start);
                    imu_cache.angular_rotation_current_y = interpolation_by_linear(imu_before_laser.angular_rotation_y, imu_after_laser.angular_rotation_y, ratio_from_start);
                    imu_cache.angular_rotation_current_z = interpolation_by_linear(imu_before_laser.angular_rotation_z, imu_after_laser.angular_rotation_z, ratio_from_start);
                }

                imu_cache.imuAngularFromStartX = imu_cache.angular_rotation_current_x - imuAngularRotationXLast;
                imu_cache.imuAngularFromStartY = imu_cache.angular_rotation_current_y - imuAngularRotationYLast;
                imu_cache.imuAngularFromStartZ = imu_cache.angular_rotation_current_z - imuAngularRotationZLast;

                imu_cache.imuAngularRotationXLast = imu_cache.angular_rotation_current_x;
                imu_cache.imuAngularRotationYLast = imu_cache.angular_rotation_current_y;
                imu_cache.imuAngularRotationZLast = imu_cache.angular_rotation_current_z;

                updateImuRollPitchYawStartSinCos();
            } else {
                vel_to_start_imu();
                transform_to_start_imu(point);
            }
        }

    }
    newest_idxIteration = newest_idx;
}

void calculate_smotthness()
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

        cloudNeighborPicked[i] = 0;
        cloud_label_[i] = 0;

        cloud_smoothness_[i].value = cloud_curvature_[i];
        cloud_smoothness_[i].ind = i;
    }
}

void mark_occluded_points()
{
    const auto &cloud_range = segmented_cloud_msg_.ground_segment_cloud_range;
    const auto &cloud_column = segmented_cloud_msg_.ground_segment_cloud_column;

    for (int i = 5; i < projected_ground_segment_cloud_->points.size() - 6; ++i) {
        float range_i = cloud_range[i];
        float range_i_1 = cloud_range[i+1];

        if (std::abs(int(cloud_column[i+1] - cloud_column[i])) < 10) {
            if (range_i - range_i_1 > 0.3) {
                cloudNeighborPicked[i - 5] = 1;
                cloudNeighborPicked[i - 4] = 1;
                cloudNeighborPicked[i - 3] = 1;
                cloudNeighborPicked[i - 2] = 1;
                cloudNeighborPicked[i - 1] = 1;
                cloudNeighborPicked[i] = 1;
            }else if (range_i_1 - range_i > 0.3) {
                cloudNeighborPicked[i + 1] = 1;
                cloudNeighborPicked[i + 2] = 1;
                cloudNeighborPicked[i + 3] = 1;
                cloudNeighborPicked[i + 4] = 1;
                cloudNeighborPicked[i + 5] = 1;
                cloudNeighborPicked[i + 6] = 1;
            }
        }

        float diff_prev = std::abs(cloud_range[i-1] - cloud_range[i]);
        float diff_next = std::abs(cloud_range[i+1] - cloud_range[i]);

        if (diff_prev > 0.02 * cloud_range[i] && diff_next > 0.02 * cloud_range[i])
            cloudNeighborPicked[i] = 1;
    }
}

void extract_features()
{
    corner_sharp_cloud_->clear();
    corner_less_sharp_cloud_->clear();
    surf_flat_cloud_->clear();
    surf_less_flat_cloud_->clear();

    const auto &cloud = projected_ground_segment_cloud_->points;
    const auto &cloud_range = segmented_cloud_msg_.ground_segment_cloud_range;
    const auto &cloud_column = segmented_cloud_msg_.ground_segment_cloud_column;

    for (int i = 0; i < N_SCAN; i++) {
        surf_less_flat_scan_->clear();

        for (int j = 0; j < 6; j++) {
            int sp = (segmented_cloud_msg_.ring_index_start[i] * (6 - j)  + segmented_cloud_msg_.ring_index_end[i] * j) / 6;
            int ep = (segmented_cloud_msg_.ring_index_start[i] * (5 - j)  + segmented_cloud_msg_.ring_index_end[i] * (j + 1)) / 6 - 1;

            if (sp >= ep)
                continue;

            std::sort(cloud_smoothness_.begin() + sp, cloud_smoothness_.begin() + ep);

            int pick_point_num = 0;
            for (int k = ep; k >= sp; k--) {
                int idx = cloud_smoothness_[k].ind;

                if (cloudNeighborPicked[idx] == 0
                    && cloud_curvature_[idx] > edgeThreshold
                    && segmented_cloud_msg_.ground_segment_flag[idx] == false) {
                
                    pick_point_num++;
                    if (pick_point_num <= 2) {
                        cloud_label_[idx] = 2;
                        corner_sharp_cloud_->push_back(cloud[idx]);
                        corner_less_sharp_cloud_->push_back(cloud[idx]);
                    } else if (pick_point_num <= 20) {
                        cloud_label_[idx] = 1;
                        corner_less_sharp_cloud_->push_back(cloud[idx]);
                    } else {
                        break;
                    }

                    cloudNeighborPicked[idx] = 1;
                    for (int l = 1; l <= 5; l++) {
                        if (std::abs(cloud_column[idx + l] - cloud_column[idx + l - 1]) > 10)
                            break;
                        cloudNeighborPicked[idx + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--) {
                        if (std::abs(cloud_column[idx + l] - cloud_column[idx + l + 1]) > 10)
                            break;
                        cloudNeighborPicked[idx + l] = 1;
                    }
                }
            }

            pick_point_num = 0;
            for (int k = sp; k <= ep; k++) {
                int idx = cloud_smoothness_[k].ind;
                if (cloudNeighborPicked[idx] == 0 &&
                    cloud_curvature_[idx] < surfThreshold &&
                    segmented_cloud_msg_.ground_segment_flag[idx] == true) {

                    cloud_label_[idx] = -1;
                    surf_flat_cloud_->push_back(projected_ground_segment_cloud_->points[idx]);

                    pick_point_num++;
                    if (pick_point_num >= 4) {
                        break;
                    }

                    cloudNeighborPicked[idx] = 1;
                    for (int l = 1; l <= 5; l++) {
                        if (std::abs(int(cloud_column[idx + l] - cloud_column[idx + l - 1])) > 10)
                            break;

                        cloudNeighborPicked[idx + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--) {
                        if (std::abs(int(cloud_column[idx + l] - cloud_column[idx + l + 1])) > 10)
                            break;

                        cloudNeighborPicked[idx + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++) {
                if (cloud_label_[k] <= 0) {
                    surf_less_flat_scan_->push_back(projected_ground_segment_cloud_->points[k]);
                }
            }
        }

        surf_less_flat_scan_ds_->clear();
        voxel_grid_filter_.setInputCloud(surf_less_flat_scan_);
        voxel_grid_filter_.filter(*surf_less_flat_scan_ds_);

        *surf_less_flat_cloud_ += *surf_less_flat_scan_ds_;
    }
}

void publish_cloud()
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

Point transform_to_start(const Point &p) const
{
    float s = 10 * (p.intensity - int(p.intensity));

    float rx = s * transformCur[0];
    float ry = s * transformCur[1];
    float rz = s * transformCur[2];
    float tx = s * transformCur[3];
    float ty = s * transformCur[4];
    float tz = s * transformCur[5];

    float x1 = std::cos(rz) * (p.x - tx) + std::sin(rz) * (p.y - ty);
    float y1 = -std::sin(rz) * (p.x - tx) + std::cos(rz) * (p.y - ty);
    float z1 = (pi->z - tz);

    float x2 = x1;
    float y2 = std::cos(rx) * y1 + std::sin(rx) * z1;
    float z2 = -std::sin(rx) * y1 + std::cos(rx) * z1;

    Point r;
    r.x = std::cos(ry) * x2 - std::sin(ry) * z2;
    r.y = y2;
    r.z = std::sin(ry) * x2 + std::cos(ry) * z2;
    r.intensity = p.intensity;

    return r;
}

void transform_to_end(Point &p)
{
    float s = 10 * (p.intensity - int(p.intensity));

    float rx = s * transformCur[0];
    float ry = s * transformCur[1];
    float rz = s * transformCur[2];
    float tx = s * transformCur[3];
    float ty = s * transformCur[4];
    float tz = s * transformCur[5];

    float x1 = std::cos(rz) * (p.x - tx) + std::sin(rz) * (p.y - ty);
    float y1 = -std::sin(rz) * (p.x - tx) + std::cos(rz) * (p.y - ty);
    float z1 = (p.z - tz);

    float x2 = x1;
    float y2 = std::cos(rx) * y1 + std::sin(rx) * z1;
    float z2 = -std::sin(rx) * y1 + std::cos(rx) * z1;

    float x3 = std::cos(ry) * x2 - std::sin(ry) * z2;
    float y3 = y2;
    float z3 = std::sin(ry) * x2 + std::cos(ry) * z2;

    rx = transformCur[0];
    ry = transformCur[1];
    rz = transformCur[2];
    tx = transformCur[3];
    ty = transformCur[4];
    tz = transformCur[5];

    float x4 = std::cos(ry) * x3 + std::sin(ry) * z3;
    float y4 = y3;
    float z4 = -std::sin(ry) * x3 + std::cos(ry) * z3;

    float x5 = x4;
    float y5 = std::cos(rx) * y4 - std::sin(rx) * z4;
    float z5 = std::sin(rx) * y4 + std::cos(rx) * z4;

    float x6 = std::cos(rz) * x5 - std::sin(rz) * y5 + tx;
    float y6 = std::sin(rz) * x5 + std::cos(rz) * y5 + ty;
    float z6 = z5 + tz;

    float x7 = roll_start_cos * (x6 - imuShiftFromStartX) 
                - roll_start_sin * (y6 - imuShiftFromStartY);
    float y7 = roll_start_sin * (x6 - imuShiftFromStartX) 
                + roll_start_cos * (y6 - imuShiftFromStartY);
    float z7 = z6 - imuShiftFromStartZ;

    float x8 = x7;
    float y8 = pitch_start_cos * y7 - pitch_start_sin * z7;
    float z8 = pitch_start_sin * y7 + pitch_start_cos * z7;

    float x9 = yaw_start_cos * x8 + yaw_start_sin * z8;
    float y9 = y8;
    float z9 = -yaw_start_sin * x8 + yaw_start_cos * z8;

    float x10 = std::cos(imuYawLast) * x9 - std::sin(imuYawLast) * z9;
    float y10 = y9;
    float z10 = std::sin(imuYawLast) * x9 + std::cos(imuYawLast) * z9;

    float x11 = x10;
    float y11 = std::cos(imuPitchLast) * y10 + std::sin(imuPitchLast) * z10;
    float z11 = -std::sin(imuPitchLast) * y10 + std::cos(imuPitchLast) * z10;

    p.x = std::cos(imuRollLast) * x11 + std::sin(imuRollLast) * y11;
    p.y = -std::sin(imuRollLast) * x11 + std::cos(imuRollLast) * y11;
    p.z = z11;
    p.intensity = int(pi->intensity);
}

void plugin_imu_rotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                        float alx, float aly, float alz, float &acx, float &acy, float &acz)
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

void accumulate_rotation(float cx, float cy, float cz, float lx, float ly, float lz, 
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

void find_corresponding_corner_features(int iterCount) {
    for (int i = 0; i < corner_sharp_cloud_->points.size(); i++) {
        auto p = transform_to_start(corner_sharp_cloud_->points[i]);
        if (iterCount % 5 == 0) {
            std::vector<int> closest_indices;
            std::vector<float> closest_square_distances;

            kdtree_last_corner_->nearestKSearch(p, 1, closest_indices, closest_square_distances);
            int minPointInd2 = -1;
            
            if (closest_square_distances[0] < nearestFeatureSearchSqDist) {
                const auto &c0 = closest_indices[0];
                int closestPointScan = int(cloud_last_corner_->points[c0].intensity);

                float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist;
                for (int j = c0 + 1; j < corner_sharp_cloud_->points.size(); j++) {
                    if (int(cloud_last_corner_->points[j].intensity) > closestPointScan + 2.5) {
                        break;
                    }
                    pointSqDis = square_distance(cloud_last_corner_->points[j], p);
                    if (int(cloud_last_corner_->points[j].intensity) > closestPointScan) {
                        if (pointSqDis < minPointSqDis2) {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                    }
                }
                for (int j = c0 - 1; j >= 0; j--) {
                    if (int(cloud_last_corner_->points[j].intensity) < closestPointScan - 2.5) {
                        break;
                    }

                    pointSqDis = square_distance(cloud_last_corner_->points[j], p);
                    if (int(cloud_last_corner_->points[j].intensity) < closestPointScan) {
                        if (pointSqDis < minPointSqDis2) {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                    }
                }
            }

            pointSearchCornerInd1[i] = c0;
            pointSearchCornerInd2[i] = minPointInd2;
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

            float a012 = sqrt(m11 * m11  + m22 * m22 + m33 * m33);

            float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

            float la =  ((y1 - y2)*m11 + (z1 - z2)*m22) / a012 / l12;

            float lb = -((x1 - x2)*m11 - (z1 - z2)*m33) / a012 / l12;

            float lc = -((x1 - x2)*m22 + (y1 - y2)*m33) / a012 / l12;

            float ld2 = a012 / l12;

            float s = 1;
            if (iterCount >= 5) {
                s = 1 - 1.8 * fabs(ld2);
            }

            if (s > 0.1 && ld2 != 0) {
                coeff_sel_->emplace_back(s * la, s * lb, s * lc, s * ld2);
                cloud_ori_->push_back(corner_sharp_cloud_->points[i]);
            }
        }
    }
}

void find_corresponding_surf_features(int iterCount) {
    for (int i = 0; i < surf_flat_cloud_->points.size(); i++) {
        auto p = transform_to_start(surf_flat_cloud_->points[i]);
        if (iterCount % 5 == 0) {
            std::vector<int> closest_indices;
            std::vector<float> closest_square_distances;

            kdtree_last_surf_->nearestKSearch(p, 1, closest_indices, closest_square_distances);
            int minPointInd2 = -1, minPointInd3 = -1;

            if (closest_square_distances[0] < nearestFeatureSearchSqDist) {
                const auto &c0 = closest_indices[0];
                int closestPointScan = int(cloud_last_surf_->points[const auto &c0].intensity);

                float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist, minPointSqDis3 = nearestFeatureSearchSqDist;
                for (int j = const auto &c0 + 1; j < surf_flat_cloud_->points.size(); j++) {
                    if (int(cloud_last_surf_->points[j].intensity) > closestPointScan + 2.5) {
                        break;
                    }

                    pointSqDis = square_distance(cloud_last_surf_->points[j], p);
                    if (int(cloud_last_surf_->points[j].intensity) <= closestPointScan) {
                        if (pointSqDis < minPointSqDis2) {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                    } else {
                        if (pointSqDis < minPointSqDis3) {
                            minPointSqDis3 = pointSqDis;
                            minPointInd3 = j;
                        }
                    }
                }
                for (int j = const auto &c0 - 1; j >= 0; j--) {
                    if (int(cloud_last_surf_->points[j].intensity) < closestPointScan - 2.5) {
                        break;
                    }

                    pointSqDis = square_distance(cloud_last_surf_->points[j], p);
                    if (int(cloud_last_surf_->points[j].intensity) >= closestPointScan) {
                        if (pointSqDis < minPointSqDis2) {
                            minPointSqDis2 = pointSqDis;
                            minPointInd2 = j;
                        }
                    } else {
                        if (pointSqDis < minPointSqDis3) {
                            minPointSqDis3 = pointSqDis;
                            minPointInd3 = j;
                        }
                    }
                }
            }

            pointSearchSurfInd1[i] = const auto &c0;
            pointSearchSurfInd2[i] = minPointInd2;
            pointSearchSurfInd3[i] = minPointInd3;
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
                coeff_sel_->emplace_back(s * pa, s * pb, s * pc, s * pd2);
                cloud_ori_->push_back(surf_flat_cloud_->points[i]);
            }
        }
    }
}

bool calculateTransformationSurf(int iterCount) {

    int pointSelNum = cloud_ori_->points.size();

    cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

    float srx = std::sin(transformCur[0]);
    float crx = std::cos(transformCur[0]);
    float sry = std::sin(transformCur[1]);
    float cry = std::cos(transformCur[1]);
    float srz = std::sin(transformCur[2]);
    float crz = std::cos(transformCur[2]);
    float tx = transformCur[3];
    float ty = transformCur[4];
    float tz = transformCur[5];

    float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
    float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
    float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

    float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz;
    float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry;

    float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
    float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

    for (int i = 0; i < pointSelNum; i++) {
        const auto &p = cloud_ori_->points[i];
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

    transformCur[0] += matX.at<float>(0, 0);
    transformCur[2] += matX.at<float>(1, 0);
    transformCur[4] += matX.at<float>(2, 0);

    for(int i=0; i<6; i++) {
        if(isnan(transformCur[i]))
            transformCur[i]=0;
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

bool calculateTransformationCorner(int iterCount) {

    int pointSelNum = cloud_ori_->points.size();

    cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

    float srx = std::sin(transformCur[0]);
    float crx = std::cos(transformCur[0]);
    float sry = std::sin(transformCur[1]);
    float cry = std::cos(transformCur[1]);
    float srz = std::sin(transformCur[2]);
    float crz = std::cos(transformCur[2]);
    float tx = transformCur[3];
    float ty = transformCur[4];
    float tz = transformCur[5];

    float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
    float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

    float c5 = crx*srz;

    for (int i = 0; i < pointSelNum; i++) {

        const auto &p = cloud_ori_->points[i];
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

    transformCur[1] += matX.at<float>(0, 0);
    transformCur[3] += matX.at<float>(1, 0);
    transformCur[5] += matX.at<float>(2, 0);

    for(int i=0; i<6; i++) {
        if(isnan(transformCur[i]))
            transformCur[i]=0;
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

bool calculateTransformation(int iterCount) {

    int pointSelNum = cloud_ori_->points.size();

    cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

    float srx = std::sin(transformCur[0]);
    float crx = std::cos(transformCur[0]);
    float sry = std::sin(transformCur[1]);
    float cry = std::cos(transformCur[1]);
    float srz = std::sin(transformCur[2]);
    float crz = std::cos(transformCur[2]);
    float tx = transformCur[3];
    float ty = transformCur[4];
    float tz = transformCur[5];

    float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
    float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
    float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

    float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
    float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

    float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
    float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

    for (int i = 0; i < pointSelNum; i++) {

        const auto &p = cloud_ori_->points[i];
        const auto &coeff = coeff_sel_->points[i];

        float arx = (-a1*p.x + a2*p.y + a3*p.z + a4) * coeff.x
                    + (a5*p.x - a6*p.y + crx*p.z + a7) * coeff.y
                    + (a8*p.x - a9*p.y - a10*p.z + a11) * coeff.z;

        float ary = (b1*p.x + b2*p.y - b3*p.z + b4) * coeff.x
                    + (b5*p.x + b6*p.y - b7*p.z + b8) * coeff.z;

        float arz = (c1*p.x + c2*p.y + c3) * coeff.x
                    + (c4*p.x - c5*p.y + c6) * coeff.y
                    + (c7*p.x + c8*p.y + c9) * coeff.z;

        float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

        float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

        float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

        float d2 = coeff.intensity;

        matA.at<float>(i, 0) = arx;
        matA.at<float>(i, 1) = ary;
        matA.at<float>(i, 2) = arz;
        matA.at<float>(i, 3) = atx;
        matA.at<float>(i, 4) = aty;
        matA.at<float>(i, 5) = atz;
        matB.at<float>(i, 0) = -0.05 * d2;
    }

    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (iterCount == 0) {
        cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

        cv::eigen(matAtA, matE, matV);
        matV.copyTo(matV2);

        is_degenerate_ = false;
        float eignThre[6] = {10, 10, 10, 10, 10, 10};
        for (int i = 5; i >= 0; i--) {
            if (matE.at<float>(0, i) < eignThre[i]) {
                for (int j = 0; j < 6; j++) {
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
        cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
        matX.copyTo(matX2);
        matX = mat_p_ * matX2;
    }

    transformCur[0] += matX.at<float>(0, 0);
    transformCur[1] += matX.at<float>(1, 0);
    transformCur[2] += matX.at<float>(2, 0);
    transformCur[3] += matX.at<float>(3, 0);
    transformCur[4] += matX.at<float>(4, 0);
    transformCur[5] += matX.at<float>(5, 0);

    for(int i=0; i<6; i++) {
        if(isnan(transformCur[i]))
            transformCur[i]=0;
    }

    float deltaR = sqrt(
                        pow(rad2deg(matX.at<float>(0, 0)), 2) +
                        pow(rad2deg(matX.at<float>(1, 0)), 2) +
                        pow(rad2deg(matX.at<float>(2, 0)), 2));
    float deltaT = sqrt(
                        pow(matX.at<float>(3, 0) * 100, 2) +
                        pow(matX.at<float>(4, 0) * 100, 2) +
                        pow(matX.at<float>(5, 0) * 100, 2));

    if (deltaR < 0.1 && deltaT < 0.1) {
        return false;
    }
    return true;
}

void check_system_initialization() {
    corner_less_sharp_cloud_.swap(cloud_last_corner_);
    surf_less_flat_cloud_.swap(cloud_last_surf_);

    kdtree_last_corner_->setInputCloud(cloud_last_corner_);
    kdtree_last_surf_->setInputCloud(cloud_last_surf_);

    sensor_msgs::PointCloud2 laserCloudCornerLast2;
    pcl::toROSMsg(*cloud_last_corner_, laserCloudCornerLast2);
    laserCloudCornerLast2.header.stamp = cloud_header_.stamp;
    laserCloudCornerLast2.header.frame_id = "camera";
    pub_last_corner_cloud_.publish(laserCloudCornerLast2);

    sensor_msgs::PointCloud2 laserCloudSurfLast2;
    pcl::toROSMsg(*cloud_last_surf_, laserCloudSurfLast2);
    laserCloudSurfLast2.header.stamp = cloud_header_.stamp;
    laserCloudSurfLast2.header.frame_id = "camera";
    pub_last_surf_cloud_.publish(laserCloudSurfLast2);

    transform_sum_[0] += imu_cache.pitch_start;
    transform_sum_[2] += imu_cache.roll_start;

    systemInitedLM = true;
}

void update_initial_guess() {
    imu_cache.imuPitchLast = imu_cache.pitch_current;
    imu_cache.imuYawLast = imu_cache.yaw_current;
    imu_cache.imuRollLast = imu_cache.roll_current;

    imu_cache.imuShiftFromStartX = imu_cache.drift_from_start_to_current_x;
    imu_cache.imuShiftFromStartY = imu_cache.drift_from_start_to_current_y;
    imu_cache.imuShiftFromStartZ = imu_cache.drift_from_start_to_current_z;

    imu_cache.imuVeloFromStartX = imu_cache.vel_diff_from_start_to_current_x;
    imu_cache.imuVeloFromStartY = imu_cache.vel_diff_from_start_to_current_y;
    imu_cache.imuVeloFromStartZ = imu_cache.vel_diff_from_start_to_current_z;

    if (imu_cache.imuAngularFromStartX != 0 || imu_cache.imuAngularFromStartY != 0 || imu_cache.imuAngularFromStartZ != 0) {
        transformCur[0] = - imu_cache.imuAngularFromStartY;
        transformCur[1] = - imu_cache.imuAngularFromStartZ;
        transformCur[2] = - imu_cache.imuAngularFromStartX;
    }
    
    if (imu_cache.imuVeloFromStartX != 0 || imu_cache.imuVeloFromStartY != 0 || imu_cache.imuVeloFromStartZ != 0) {
        transformCur[3] -= imu_cache.imuVeloFromStartX * scanPeriod;
        transformCur[4] -= imu_cache.imuVeloFromStartY * scanPeriod;
        transformCur[5] -= imu_cache.imuVeloFromStartZ * scanPeriod;
    }
}

void update_transformation() {
    if (cloud_last_corner_->points.size() < 10 || cloud_last_surf_->points.size() < 100)
        return;

    for (int iterCount1 = 0; iterCount1 < 25; iterCount1++) {
        cloud_ori_->clear();
        coeff_sel_->clear();

        find_corresponding_surf_features(iterCount1);

        if (cloud_ori_->points.size() < 10)
            continue;
        if (calculateTransformationSurf(iterCount1) == false)
            break;
    }

    for (int iterCount2 = 0; iterCount2 < 25; iterCount2++) {

        cloud_ori_->clear();
        coeff_sel_->clear();

        find_corresponding_corner_features(iterCount2);

        if (cloud_ori_->points.size() < 10)
            continue;
        if (calculateTransformationCorner(iterCount2) == false)
            break;
    }
}

void integrate_transformation() {
    float rx, ry, rz, tx, ty, tz;
    accumulate_rotation(transform_sum_[0], transform_sum_[1], transform_sum_[2], 
                        -transformCur[0], -transformCur[1], -transformCur[2], rx, ry, rz);

    float x1 = std::cos(rz) * (transformCur[3] - imu_cache.imuShiftFromStartX) 
                - std::sin(rz) * (transformCur[4] - imu_cache.imuShiftFromStartY);
    float y1 = std::sin(rz) * (transformCur[3] - imu_cache.imuShiftFromStartX) 
                + std::cos(rz) * (transformCur[4] - imu_cache.imuShiftFromStartY);
    float z1 = transformCur[5] - imu_cache.imuShiftFromStartZ;

    float x2 = x1;
    float y2 = std::cos(rx) * y1 - std::sin(rx) * z1;
    float z2 = std::sin(rx) * y1 + std::cos(rx) * z1;

    tx = transform_sum_[3] - (std::cos(ry) * x2 + std::sin(ry) * z2);
    ty = transform_sum_[4] - y2;
    tz = transform_sum_[5] - (-std::sin(ry) * x2 + std::cos(ry) * z2);

    plugin_imu_rotation(rx, ry, rz, pitch_start, yaw_start, roll_start, 
                        imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);

    transform_sum_[0] = rx;
    transform_sum_[1] = ry;
    transform_sum_[2] = rz;
    transform_sum_[3] = tx;
    transform_sum_[4] = ty;
    transform_sum_[5] = tz;
}

void publish_odometry() {
    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transform_sum_[2], -transform_sum_[0], -transform_sum_[1]);

    laser_odometry_.header.stamp = cloud_header_.stamp;
    laser_odometry_.pose.pose.orientation.x = -geoQuat.y;
    laser_odometry_.pose.pose.orientation.y = -geoQuat.z;
    laser_odometry_.pose.pose.orientation.z = geoQuat.x;
    laser_odometry_.pose.pose.orientation.w = geoQuat.w;
    laser_odometry_.pose.pose.position.x = transform_sum_[3];
    laser_odometry_.pose.pose.position.y = transform_sum_[4];
    laser_odometry_.pose.pose.position.z = transform_sum_[5];
    pub_laser_odometry_.publish(laser_odometry_);

    laser_odometry_trans_.stamp_ = cloud_header_.stamp;
    laser_odometry_trans_.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
    laser_odometry_trans_.setOrigin(tf::Vector3(transform_sum_[3], transform_sum_[4], transform_sum_[5]));
    tf_broadcaster_.sendTransform(laser_odometry_trans_);
}

void adjust_outlier_cloud() {
    for (auto &p : projected_outlier_cloud_->points)
    {
        std::array<float, 3> tmp = {p.x, p.y, p.z};
        point.x = tmp[1];
        point.y = tmp[2];
        point.z = tmp[0];
    }
}

void publish_cloud_last() {
    updateImuRollPitchYawStartSinCos();

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

    frame_count_++;

    if (frame_count_ >= skip_frame_num_ + 1) {
        frame_count_ = 0;

        adjust_outlier_cloud();
        sensor_msgs::PointCloud2 outlierCloudLast2;
        pcl::toROSMsg(*projected_outlier_cloud_, outlierCloudLast2);
        outlierCloudLast2.header.stamp = cloud_header_.stamp;
        outlierCloudLast2.header.frame_id = "camera";
        pub_last_outlier_cloud_.publish(outlierCloudLast2);

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*cloud_last_corner_, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = cloud_header_.stamp;
        laserCloudCornerLast2.header.frame_id = "camera";
        pub_last_corner_cloud_.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*cloud_last_surf_, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = cloud_header_.stamp;
        laserCloudSurfLast2.header.frame_id = "camera";
        pub_last_surf_cloud_.publish(laserCloudSurfLast2);
    }
}

void run()
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

    calculate_smotthness();

    mark_occluded_points();

    extract_features();

    // cloud for visualization
    publish_cloud();

    /**
    2. Feature Association
    */
    if (!systemInitedLM) {
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
