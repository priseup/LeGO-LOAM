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

#ifndef LEGO_FEATURE_ASSOCATION_H_
#define LEGO_FEATURE_ASSOCATION_H_

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include <array>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <Eigen/Dense>

#include "cloud_msgs/cloud_info.h"
#include "utility.h"

class FeatureAssociation{
public:
    FeatureAssociation();
    void run();

private:
    void init();

    Point transform_to_start(const Point &p);
    void transform_to_end(Point &p);

    void check_system_initialization();

    void calculate_transformation();

    void publish_odometry();

    void publish_cloud_last();

/*
    void update_imu_rotation_start_sin_cos();
    void shift_to_start_imu(const float &point_time);
    void vel_to_start_imu();
    void transform_to_start_imu(Point &p);
*/

    void imu_handler(const sensor_msgs::Imu::ConstPtr& imuIn);

    void laser_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& laser_cloud);

    void outlier_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& msgIn);

    void laser_cloud_msg_handler(const cloud_msgs::cloud_infoConstPtr& msgIn);
    
    void adjust_distortion();
    
    void calculate_smoothness();

    void mark_occluded_points();

    void extract_features();
    
    void publish_cloud();

    void reset_parameters();

    void mark_neibor_is_picked(int idx);

    std::array<int, 2> find_closest_in_same_adjacent_ring(int closest_idx, const Point &p, const pcl::PointCloud<Point>::Ptr &cloud, bool get_same);
    int point_scan_id(const Point &p);

private:
    enum class FeatureLabel
    {
        surf_flat,
        surf_less_flat,
        corner_sharp,
        corner_less_sharp
    };

    struct ImuFrame
    {
        double time = 0.0;

        float roll = 0.f;
        float pitch = 0.f;
        float yaw = 0.f;

        float acc_x = 0.f;
        float acc_y = 0.f;
        float acc_z = 0.f;

        float vel_x = 0.f;
        float vel_y = 0.f;
        float vel_z = 0.f;

        float shift_x = 0.f;
        float shift_y = 0.f;
        float shift_z = 0.f;

        float angular_vel_x = 0.f;
        float angular_vel_y = 0.f;
        float angular_vel_z = 0.f;

        float angular_x = 0.f;
        float angular_y = 0.f;
        float angular_z = 0.f;
    };
    struct ImuFrame imus_[imuQueLength];
    int imu_idx_after_laser_ = 0; // first imu newer than laser point time
    int imu_idx_new_ = -1;  // newest imu in imus_
    int imu_idx_last_used_ = 0; // idx of used imu in imus_

    struct smoothness_t{ 
        float value;
        int idx;

        bool operator < (const smoothness_t &other) const
        {
            return value < other.value;
        }
    };
private:
    ros::NodeHandle nh_;

    ros::Subscriber sub_laser_cloud_;
    ros::Subscriber sub_laser_cloud_info_;
    ros::Subscriber sub_outlier_cloud_;
    ros::Subscriber sub_imu_;

    ros::Publisher pub_corner_sharp_;
    ros::Publisher pub_corner_less_sharp_;
    ros::Publisher pub_surf_flat_;
    ros::Publisher pub_surf_less_flat_;

    ros::Publisher pub_last_corner_cloud_;
    ros::Publisher pub_last_surf_cloud_;
    ros::Publisher pub_laser_odometry_;
    ros::Publisher pub_last_outlier_cloud_;

    pcl::PointCloud<Point>::Ptr projected_ground_segment_cloud_;
    pcl::PointCloud<Point>::Ptr projected_outlier_cloud_;

    pcl::PointCloud<Point>::Ptr corner_sharp_cloud_;
    pcl::PointCloud<Point>::Ptr corner_less_sharp_cloud_;
    pcl::PointCloud<Point>::Ptr surf_flat_cloud_;
    pcl::PointCloud<Point>::Ptr surf_less_flat_cloud_;

    pcl::PointCloud<Point>::Ptr cloud_last_corner_;
    pcl::PointCloud<Point>::Ptr cloud_last_surf_;

    pcl::KdTreeFLANN<Point>::Ptr kd_last_corner_;
    pcl::KdTreeFLANN<Point>::Ptr kd_last_surf_;

    double laser_scan_time_ = 0;
    double segment_cloud_time_ = 0;
    double segment_cloud_info_time_ = 0;
    double outlier_cloud_time_ = 0;

    bool has_get_cloud_ = false;
    bool has_get_cloud_msg_ = false;
    bool has_get_outlier_cloud_ = false;

    cloud_msgs::cloud_info segmented_cloud_msg_;
    std_msgs::Header cloud_header_;

    std::vector<smoothness_t> cloud_smoothness_;
    std::vector<float> cloud_curvature_;
    std::vector<int> is_neibor_picked_;
    std::vector<FeatureLabel> cloud_label_;

    bool is_system_inited_ = false;

    double pose_params_[6] = {0, 0, 0, 0, 0, 0}; // tx, ty, tz, roll, pitch, yaw, current frame to last frame
    Eigen::Matrix3d r_w_curr_; // rotation of current lidar frame to world frame
    Eigen::Vector3d t_w_curr_; // translation of current lidar frame to world frame

    Eigen::Quaterniond q_last_curr_; // rotation of current lidar frame to last lidar frame
    Eigen::Vector3d t_last_curr_; // translation of current lidar frame to last lidar frame

    // Eigen::Map<Eigen::Vector3d> t_last_curr_(pose_params_ + 3);
    // Eigen::Map<Eigen::Quaterniond> q_last_curr_(pose_params_);


    nav_msgs::Odometry laser_odom_;
    tf::TransformBroadcaster tf_broad_;
};

#endif  // LEGO_FEATURE_ASSOCIATION_H_
