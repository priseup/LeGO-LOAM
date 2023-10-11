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

#include "utility.h"

class FeatureAssociation{
public:
    FeatureAssociation();
    void run();

private:
    void init();

    Point transform_to_start(const Point &p);
    void transform_to_end(Point &p)

    void plugin_imu_rotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                           float alx, float aly, float alz, float &acx, float &acy, float &acz);

    void accumulate_rotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                            float &ox, float &oy, float &oz);

    void find_corresponding_corner_features();
    void find_corresponding_surf_features();

    bool calculate_suf_transformation(int iterCount);

    bool calculate_corner_transformation(int iterCount);

    bool calculate_transformation(int iterCount);

    void check_system_initialization();

    void update_initial_guess();

    void update_transformation();

    void integrate_transformation(); 

    void publish_odometry();

    void adjust_outlier_cloud(); 

    void publish_cloud_last();

    void update_imu_rotation_start_sin_cos();

    void shift_to_start_imu(float point_time);

    void vel_to_start_imu();

    void transform_to_start_imu(Point &p);

    void accumulate_imu_shift_rotation();

    void imu_handler(const sensor_msgs::Imu::ConstPtr& imuIn);

    void laser_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& laser_cloud);

    void outlier_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& msgIn);

    void laser_cloud_msg_handler(const cloud_msgs::cloud_infoConstPtr& msgIn);
    
    void adjust_distortion();
    
    void calculate_smotthness();

    void mark_occluded_points();

    void extract_features();
    
    void publish_cloud();

    void mark_neibor_is_picked(int idx);

    int find_closest_in_adjacent_ring(int closest_idx, const Point &p, const pcl::PointCloud<Point>::Ptr &cloud);
    std::array<float, 2> FeatureAssociation::find_closest_in_same_adjacent_ring(int closest_idx, const Point &p, const pcl::PointCloud<Point>::Ptr &cloud);
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

        float angular_rotation_x = 0.f;
        float angular_rotation_y = 0.f;
        float angular_rotation_z = 0.f;
    };
    struct ImuCache
    {
        int after_laser_idx = 0; // first imu newer than laser point time
        int newest_idx = -1;
        int last_new_idx = 0;

        float roll_start = 0.f;
        float pitch_start = 0.f;
        float yaw_start = 0.f;;

        float roll_start_cos = 0.f;
        float roll_start_sin = 0.f;
        float pitch_start_cos = 0.f;
        float pitch_start_sin = 0.f;
        float yaw_start_cos = 0.f;
        float yaw_start_sin = 0.f;

        float vel_start_x = 0.f;
        float vel_start_y = 0.f;
        float vel_start_z = 0.f;

        float shift_start_x = 0.f;
        float shift_start_y = 0.f;
        float shift_start_z = 0.f;

        float roll_current = 0.f;
        float pitch_current = 0.f;
        float yaw_current = 0.f;

        float vel_current_x = 0.f;
        float vel_current_y = 0.f;
        float vel_current_z = 0.f;

        float shift_current_x = 0.f;
        float shift_current_y = 0.f;
        float shift_current_z = 0.f;

        float drift_from_start_to_current_x = 0.f;
        float drift_from_start_to_current_y = 0.f;
        float drift_from_start_to_current_z = 0.f;

        float vel_diff_from_start_to_current_x = 0.f;
        float vel_diff_from_start_to_current_y = 0.f;
        float vel_diff_from_start_to_current_z = 0.f;

        float angular_rotation_current_x = 0.f;
        float angular_rotation_current_y = 0.f;
        float angular_rotation_current_z = 0.f;

        float last_angular_rotation_x = 0.f;
        float last_angular_rotation_y = 0.f;
        float last_angular_rotation_z = 0.f;

        float angular_diff_from_start_to_current_x = 0.f;
        float angular_diff_from_start_to_current_y = 0.f;
        float angular_diff_from_start_to_current_z = 0.f;

        struct ImuFrame imu_queue[imuQueLength];

        int idx_increment(int idx) const
        {
            return (idx + 1) % imuQueLength;
        }
        int idx_decrement(int idx) const
        {
            return (idx + imuQueLength - 1) % imuQueLength;
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
    pcl::PointCloud<Point>::Ptr cloud_ori_;
    pcl::PointCloud<Point>::Ptr coeff_sel_;

    pcl::KdTreeFLANN<Point>::Ptr kdtree_last_corner_;
    pcl::KdTreeFLANN<Point>::Ptr kdtree_last_surf_;

    pcl::VoxelGrid<Point> voxel_grid_filter_;

    struct ImuCache imu_cache;

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
    float *cloud_curvature_;
    int *is_neibor_picked_;
    std::vector<FeatureLabel> cloud_label_;

    int frame_count_ = 1;
    int skip_frame_num_ = 1;
    bool is_system_inited_ = false;

    float *pointSearchCornerInd1;
    float *pointSearchCornerInd2;

    float *pointSearchSurfInd1;
    float *pointSearchSurfInd2;
    float *pointSearchSurfInd3;

    float transformCur[6];
    float transform_sum_[6];

    nav_msgs::Odometry laser_odometry_;

    tf::TransformBroadcaster tf_broadcaster_;
    tf::StampedTransform laser_odometry_trans_;

    bool is_degenerate_ = false;
    cv::Mat mat_p_;
};
