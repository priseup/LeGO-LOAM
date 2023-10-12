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
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <memory>
#include <unordered_set>
#include "utility.h"
#include "lego_math.h"

using namespace gtsam;

class mapOptimization{
private:
    struct ImuFrame
    {
        double time = 0;
        float roll = 0;
        float pitch = 0;
    };

    struct ImuCache
    {
        int after_laser_idx = 0; // first imu newer than laser frame
        int newest_idx = -1;

        struct ImuFrame imu[imuQueLength];

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
    gtsam::NonlinearFactorGraph gtsam_graph_;
    gtsam::Values initial_estimate_;
    gtsam::Values optimized_estimate_;
    gtsam::Values current_estimate_;
    std::unique_ptr<gtsam::ISAM2> isam_;

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odometry_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr constraint_noise_;

    ros::NodeHandle nh_;

    ros::Publisher pub_laser_cloud_surround_;
    ros::Publisher pub_odom_after_mapped_;
    ros::Publisher pub_key_poses_;

    ros::Publisher pub_history_key_frames_;
    ros::Publisher pub_icp_key_frames_;
    ros::Publisher pub_recent_key_frames_;
    ros::Publisher pub_registered_cloud_;

    ros::Subscriber sub_corner_last_;
    ros::Subscriber sub_surf_last_;
    ros::Subscriber sub_outlier_last_;
    ros::Subscriber sub_laser_odom_;
    ros::Subscriber sub_imu_;

    nav_msgs::Odometry odom_mapped_;
    tf::StampedTransform transform_mapped_;
    tf::TransformBroadcaster tf_broadcaster_;

    vector<pcl::PointCloud<Point>::Ptr> corner_key_frames_;
    vector<pcl::PointCloud<Point>::Ptr> surf_key_frames_;
    vector<pcl::PointCloud<Point>::Ptr> recent_key_frames_;

    deque<pcl::PointCloud<Point>::Ptr> recent_corner_key_frames_;
    deque<pcl::PointCloud<Point>::Ptr> recent_surf_key_frames_;
    deque<pcl::PointCloud<Point>::Ptr> recent_outlier_key_frames_;
    int latest_frame_id_ = 0;

    // std::unordered_set<int> surrounding_existing_key_poses_id_;
    std::vector<int> surrounding_existing_key_poses_id_;
    deque<pcl::PointCloud<Point>::Ptr> surrounding_corner_key_frames_;
    deque<pcl::PointCloud<Point>::Ptr> surrounding_surf_key_frames_;
    deque<pcl::PointCloud<Point>::Ptr> surrounding_outlier_key_frames_;
    
    Point prev_pose_;
    Point current_pose_;

    pcl::PointCloud<Point>::Ptr key_poses_3d_;
    pcl::PointCloud<PointPose>::Ptr key_poses_6d_;

    pcl::PointCloud<Point>::Ptr surrounding_key_poses_;
    pcl::PointCloud<Point>::Ptr surrounding_key_poses_ds_;

    pcl::PointCloud<Point>::Ptr cloud_last_corner_; // corner feature set from odoOptimization
    pcl::PointCloud<Point>::Ptr cloud_last_surf_; // surf feature set from odoOptimization
    pcl::PointCloud<Point>::Ptr last_corner_ds_; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<Point>::Ptr last_surf_ds_; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<Point>::Ptr cloud_last_outlier_; // corner feature set from odoOptimization
    pcl::PointCloud<Point>::Ptr last_outlier_ds_; // corner feature set from odoOptimization

    pcl::PointCloud<Point>::Ptr last_total_surf_; // surf feature set from odoOptimization
    pcl::PointCloud<Point>::Ptr last_total_surf_ds_; // downsampled corner featuer set from odoOptimization

    pcl::PointCloud<Point>::Ptr cloud_ori_;
    pcl::PointCloud<Point>::Ptr coeff_sel_;

    pcl::PointCloud<Point>::Ptr corner_map_;
    pcl::PointCloud<Point>::Ptr surf_map_;
    pcl::PointCloud<Point>::Ptr corner_map_ds_;
    pcl::PointCloud<Point>::Ptr surf_map_ds_;

    pcl::KdTreeFLANN<Point>::Ptr kdtree_corner_map_;
    pcl::KdTreeFLANN<Point>::Ptr kdtree_surf_map_;

    pcl::KdTreeFLANN<Point>::Ptr kdtree_surrounding_key_poses_;
    pcl::KdTreeFLANN<Point>::Ptr kdtree_history_key_poses_;
    
    pcl::PointCloud<Point>::Ptr near_history_surf_key_frame_;
    pcl::PointCloud<Point>::Ptr near_history_surf_key_frame_ds_;

    pcl::PointCloud<Point>::Ptr lastest_corner_key_frame_;
    pcl::PointCloud<Point>::Ptr lastest_surf_key_frame_;
    pcl::PointCloud<Point>::Ptr lastest_surf_key_frame_ds_;

    pcl::KdTreeFLANN<Point>::Ptr kdtree_global_map_;
    pcl::PointCloud<Point>::Ptr global_map_key_poses_;
    pcl::PointCloud<Point>::Ptr global_map_key_poses_ds_;
    pcl::PointCloud<Point>::Ptr global_map_key_frames_;
    pcl::PointCloud<Point>::Ptr global_map_key_frames_ds_;

    std::vector<int> closest_indices_;
    std::vector<float> closest_square_distances_;

    pcl::VoxelGrid<Point> vg_corner_filter_;
    pcl::VoxelGrid<Point> vg_surf_filter_;
    pcl::VoxelGrid<Point> vg_outlier_filter_;
    pcl::VoxelGrid<Point> vg_history_key_frames_filter_; // for histor key frames of loop closure
    pcl::VoxelGrid<Point> vg_surrounding_key_poses_filter_; // for surrounding key poses of scan-to-map optimization
    pcl::VoxelGrid<Point> vg_global_map_key_poses_filter_; // for global map visualization
    pcl::VoxelGrid<Point> vg_global_map_key_frames_filter_; // for global map visualization

    double time_corner_last_ = 0;
    double time_surf_last_ = 0;
    double time_outlier_last_ = 0;
    double time_odom_ = 0;

    bool has_get_corner_last_ = false;
    bool has_get_surf_last_ = false;
    bool has_get_outlier_last_ = false;
    bool has_get_laser_odom_ = false;

    float transform_last_[6];
    float transform_sum_[6];
    float transformIncre[6];
    float transformTobeMapped[6];
    float transformBefMapped[6];
    float transformAftMapped[6];

    struct ImuCache imu_cache;

    std::mutex mutex_;

    double time_last_process_ = -1;

    cv::Mat matA0;
    cv::Mat matB0;
    cv::Mat matX0;

    cv::Mat matA1;
    cv::Mat matD1;
    cv::Mat matV1;

    bool is_degenerate_ = false;
    cv::Mat mat_p_;

    int closest_history_frame_id_;
    int lastest_frame_id_loop_closure_;

    bool is_closure_loop_ = false;

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

public:
    mapOptimization(): nh_("~")
    {
    	gtsam::ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.01;
		parameters.relinearizeSkip = 1;
    	isam_->reset(new ISAM2(parameters));

        pub_key_poses_ = nh_.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pub_laser_cloud_surround_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        pub_odom_after_mapped_ = nh_.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);

        sub_corner_last_ = nh_.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        sub_surf_last_ = nh_.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        sub_outlier_last_ = nh_.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2, &mapOptimization::laserCloudOutlierLastHandler, this);
        sub_laser_odom_ = nh_.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laser_odom_handler, this);
        sub_imu_ = nh_.subscribe<sensor_msgs::Imu> (imuTopic, 50, &mapOptimization::imu_handler, this);

        pub_history_key_frames_ = nh_.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pub_icp_key_frames_ = nh_.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pub_recent_key_frames_ = nh_.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
        pub_registered_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/registered_cloud", 2);

        vg_corner_filter_.setLeafSize(0.2, 0.2, 0.2);
        vg_surf_filter_.setLeafSize(0.4, 0.4, 0.4);
        vg_outlier_filter_.setLeafSize(0.4, 0.4, 0.4);

        vg_history_key_frames_filter_.setLeafSize(0.4, 0.4, 0.4); // for histor key frames of loop closure
        vg_surrounding_key_poses_filter_.setLeafSize(1.0, 1.0, 1.0); // for surrounding key poses of scan-to-map optimization

        vg_global_map_key_poses_filter_.setLeafSize(1.0, 1.0, 1.0); // for global map visualization
        vg_global_map_key_frames_filter_.setLeafSize(0.4, 0.4, 0.4); // for global map visualization

        odom_mapped_.header.frame_id = "camera_init";
        odom_mapped_.child_frame_id = "aft_mapped";

        transform_mapped_.frame_id_ = "camera_init";
        transform_mapped_.child_frame_id_ = "aft_mapped";

        allocate_memory();
    }

    void allocate_memory() {
        key_poses_3d_.reset(new pcl::PointCloud<Point>);
        key_poses_6d_.reset(new pcl::PointCloud<PointPose>());

        kdtree_surrounding_key_poses_.reset(new pcl::KdTreeFLANN<Point>);
        kdtree_history_key_poses_.reset(new pcl::KdTreeFLANN<Point>);

        surrounding_key_poses_.reset(new pcl::PointCloud<Point>);
        surrounding_key_poses_ds_.reset(new pcl::PointCloud<Point>);        

        cloud_last_corner_.reset(new pcl::PointCloud<Point>); // corner feature set from odoOptimization
        cloud_last_surf_.reset(new pcl::PointCloud<Point>); // surf feature set from odoOptimization
        last_corner_ds_.reset(new pcl::PointCloud<Point>); // downsampled corner featuer set from odoOptimization
        last_surf_ds_.reset(new pcl::PointCloud<Point>); // downsampled surf featuer set from odoOptimization
        cloud_last_outlier_.reset(new pcl::PointCloud<Point>); // corner feature set from odoOptimization
        last_outlier_ds_.reset(new pcl::PointCloud<Point>); // downsampled corner feature set from odoOptimization
        last_total_surf_.reset(new pcl::PointCloud<Point>); // surf feature set from odoOptimization
        last_total_surf_ds_.reset(new pcl::PointCloud<Point>); // downsampled surf featuer set from odoOptimization

        cloud_ori_.reset(new pcl::PointCloud<Point>);
        coeff_sel_.reset(new pcl::PointCloud<Point>);

        corner_map_.reset(new pcl::PointCloud<Point>);
        surf_map_.reset(new pcl::PointCloud<Point>);
        corner_map_ds_.reset(new pcl::PointCloud<Point>);
        surf_map_ds_.reset(new pcl::PointCloud<Point>);

        kdtree_corner_map_.reset(new pcl::KdTreeFLANN<Point>);
        kdtree_surf_map_.reset(new pcl::KdTreeFLANN<Point>);
        
        near_history_surf_key_frame_.reset(new pcl::PointCloud<Point>);
        near_history_surf_key_frame_ds_.reset(new pcl::PointCloud<Point>);

        lastest_corner_key_frame_.reset(new pcl::PointCloud<Point>);
        lastest_surf_key_frame_.reset(new pcl::PointCloud<Point>);
        lastest_surf_key_frame_ds_.reset(new pcl::PointCloud<Point>);

        kdtree_global_map_.reset(new pcl::KdTreeFLANN<Point>);
        global_map_key_poses_.reset(new pcl::PointCloud<Point>);
        global_map_key_poses_ds_.reset(new pcl::PointCloud<Point>);
        global_map_key_frames_.reset(new pcl::PointCloud<Point>);
        global_map_key_frames_ds_.reset(new pcl::PointCloud<Point>);

        for (int i = 0; i < 6; ++i) {
            transform_last_[i] = 0;
            transform_sum_[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        // for (int i = 0; i < imuQueLength; ++i) {
        //     imuTime[i] = 0;
        //     imuRoll[i] = 0;
        //     imuPitch[i] = 0;
        // }
        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        prior_noise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
        odometry_noise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);

        matA0 = cv::Mat(5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat(5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat(3, 1, CV_32F, cv::Scalar::all(0));

        matA1 = cv::Mat(3, 3, CV_32F, cv::Scalar::all(0));
        matD1 = cv::Mat(1, 3, CV_32F, cv::Scalar::all(0));
        matV1 = cv::Mat(3, 3, CV_32F, cv::Scalar::all(0));

        mat_p_ = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    void transformAssociateToMap()
    {
        float x1 = std::cos(transform_sum_[1]) * (transformBefMapped[3] - transform_sum_[3]) 
                 - std::sin(transform_sum_[1]) * (transformBefMapped[5] - transform_sum_[5]);
        float y1 = transformBefMapped[4] - transform_sum_[4];
        float z1 = std::sin(transform_sum_[1]) * (transformBefMapped[3] - transform_sum_[3]) 
                 + std::cos(transform_sum_[1]) * (transformBefMapped[5] - transform_sum_[5]);

        float x2 = x1;
        float y2 = std::cos(transform_sum_[0]) * y1 + std::sin(transform_sum_[0]) * z1;
        float z2 = -std::sin(transform_sum_[0]) * y1 + std::cos(transform_sum_[0]) * z1;

        transformIncre[3] = std::cos(transform_sum_[2]) * x2 + std::sin(transform_sum_[2]) * y2;
        transformIncre[4] = -std::sin(transform_sum_[2]) * x2 + std::cos(transform_sum_[2]) * y2;
        transformIncre[5] = z2;

        float sbcx = std::sin(transform_sum_[0]);
        float cbcx = std::cos(transform_sum_[0]);
        float sbcy = std::sin(transform_sum_[1]);
        float cbcy = std::cos(transform_sum_[1]);
        float sbcz = std::sin(transform_sum_[2]);
        float cbcz = std::cos(transform_sum_[2]);

        float sblx = std::sin(transformBefMapped[0]);
        float cblx = std::cos(transformBefMapped[0]);
        float sbly = std::sin(transformBefMapped[1]);
        float cbly = std::cos(transformBefMapped[1]);
        float sblz = std::sin(transformBefMapped[2]);
        float cblz = std::cos(transformBefMapped[2]);

        float salx = std::sin(transformAftMapped[0]);
        float calx = std::cos(transformAftMapped[0]);
        float saly = std::sin(transformAftMapped[1]);
        float caly = std::cos(transformAftMapped[1]);
        float salz = std::sin(transformAftMapped[2]);
        float calz = std::cos(transformAftMapped[2]);

        float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                  - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                  - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                  - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                     - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                     - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                     + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                     + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
        float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                     - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                     + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                     + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                     - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                     + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
        transformTobeMapped[1] = std::atan2(srycrx / std::cos(transformTobeMapped[0]), 
                                       crycrx / std::cos(transformTobeMapped[0]));
        
        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                     + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        transformTobeMapped[2] = std::atan2(srzcrx / std::cos(transformTobeMapped[0]), 
                                       crzcrx / std::cos(transformTobeMapped[0]));

        x1 = std::cos(transformTobeMapped[2]) * transformIncre[3] - std::sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = std::sin(transformTobeMapped[2]) * transformIncre[3] + std::cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = std::cos(transformTobeMapped[0]) * y1 - std::sin(transformTobeMapped[0]) * z1;
        z2 = std::sin(transformTobeMapped[0]) * y1 + std::cos(transformTobeMapped[0]) * z1;

        transformTobeMapped[3] = transformAftMapped[3] 
                               - (std::cos(transformTobeMapped[1]) * x2 + std::sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5] 
                               - (-std::sin(transformTobeMapped[1]) * x2 + std::cos(transformTobeMapped[1]) * z2);
    }

    void transform_update()
    {
		if (imu_cache.newest_idx >= 0) {
		    float imuRollLast = 0, imuPitchLast = 0;
		    while (imu_cache.after_laser_idx != newest_idx) {
		        if (time_odom_ + scanPeriod < imu_cache.imu[imu_cache.after_laser_idx].time) {
		            break;
		        }
		        imu_cache.after_laser_idx = imu_cache.idx_increment(imu_cache.after_laser_idx);
		    }

            const auto &imu_after_laser = imu_cache.imu[imu_cache.after_laser_idx];
		    if (time_odom_ + scanPeriod > imu_after_laser.time) {
		        imuRollLast = imu_after_laser.roll;
		        imuPitchLast = imu_after_laser.pitch;
		    } else {
                int before_laser_idx = imu_cache.idx_decrement(imu_cache.after_laser_idx);
                const auto &imu_before_laser = imu_cache.imu_queue[before_laser_idx];
                float ratio_from_start = (time_odom_ + scanPeriod - imu_before_laser.time) 
                imuRollLast = interpolation_by_linear(imu_before_laser.roll, imu_after_laser.roll, ratio_from_start);
                imuPitchLast = interpolation_by_linear(imu_before_laser.pitch, imu_after_laser.pitch, ratio_from_start);
		    }

		    transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
		    transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
		  }

		for (int i = 0; i < 6; i++) {
		    transformBefMapped[i] = transform_sum_[i];
		    transformAftMapped[i] = transformTobeMapped[i];
		}
    }

    void updatePointAssociateToMapSinCos() {
        cRoll = std::cos(transformTobeMapped[0]);
        sRoll = std::sin(transformTobeMapped[0]);

        cPitch = std::cos(transformTobeMapped[1]);
        sPitch = std::sin(transformTobeMapped[1]);

        cYaw = std::cos(transformTobeMapped[2]);
        sYaw = std::sin(transformTobeMapped[2]);

        tX = transformTobeMapped[3];
        tY = transformTobeMapped[4];
        tZ = transformTobeMapped[5];
    }

    Point pointAssociateToMap(const Point &p)
    {
        auto r = rotate_by_zxy(pi.x, pi.y, pi.z,
                                cRoll, sRoll,
                                cPitch, sPitch,
                                cYaw, sYaw);

        Point po;
        po.x = r[0] + tX;
        po.y = r[1] + tY;
        po.z = r[2] + tZ;
        po.intensity = pi.intensity;

        return po;
    }

    void update_transform_sin_cos(const PointPose &p) {
        ctRoll = std::cos(p.roll);
        stRoll = std::sin(p.roll);

        ctPitch = std::cos(p.pitch);
        stPitch = std::sin(p.pitch);

        ctYaw = std::cos(p.yaw);
        stYaw = std::sin(p.yaw);

        tInX = p.x;
        tInY = p.y;
        tInZ = p.z;
    }

    pcl::PointCloud<Point>::Ptr transformPointCloud(pcl::PointCloud<Point>::Ptr cloudIn) {
	// !!! DO NOT use pcl for point cloud transformation, results are not accurate
        // Reason: unkown
        pcl::PointCloud<Point>::Ptr cloudOut(new pcl::PointCloud<Point>);
        cloudOut->resize(cloudIn->points.size());

        for (int i = 0; i < cloudIn->points.size(); ++i) {
            const auto &p = cloudIn->points[i];
            auto &po = cloudOut->points[i];

            auto r = rotate_by_zxy(p.x, p.y, p.z,
                                ctRoll, stRoll,
                                ctPitch, stPitch,
                                ctYaw, stYaw);

            po.x = r[0] + tInX;
            po.y = r[1] + tInY;
            po.z = r[2] + tInZ;
            po.intensity = p.intensity;
        }
        return cloudOut;
    }

    pcl::PointCloud<Point>::Ptr transformPointCloud(pcl::PointCloud<Point>::Ptr cloudIn, PointPose* transformIn) {
        pcl::PointCloud<Point>::Ptr cloudOut(new pcl::PointCloud<Point>);
        cloudOut->resize(cloudIn->points.size());
        
        for (int i = 0; i < cloudIn->points.size(); ++i) {
            const auto &p = cloudIn->points[i];
            auto &po = cloudOut->points[i];

            auto r = rotate_by_zxy(p.x, p.y, p.z,
                                transformIn->roll,
                                transformIn->pitch,
                                transformIn->yaw);

            po.x = r[0] + transformIn->x;
            po.y = r[1] + transformIn->y;
            po.z = r[2] + transformIn->z;
            po.intensity = p.intensity;
        }
        return cloudOut;
    }

    void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg) {
        time_outlier_last_ = msg->header.stamp.toSec();
        pcl::fromROSMsg(*msg, *cloud_last_outlier_);
        has_get_outlier_last_ = true;
    }

    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg) {
        time_corner_last_ = msg->header.stamp.toSec();
        pcl::fromROSMsg(*msg, *cloud_last_corner_);
        has_get_corner_last_ = true;
    }

    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg) {
        time_surf_last_ = msg->header.stamp.toSec();
        pcl::fromROSMsg(*msg, *cloud_last_surf_);
        has_get_surf_last_ = true;
    }

    void laser_odom_handler(const nav_msgs::Odometry::ConstPtr &odom) {
        time_odom_ = odom->header.stamp.toSec();
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = odom->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        transform_sum_[0] = -pitch;
        transform_sum_[1] = -yaw;
        transform_sum_[2] = roll;
        transform_sum_[3] = odom->pose.pose.position.x;
        transform_sum_[4] = odom->pose.pose.position.y;
        transform_sum_[5] = odom->pose.pose.position.z;
        has_get_laser_odom_ = true;
    }

    void imu_handler(const sensor_msgs::Imu::ConstPtr& imuIn) {
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        imu_cache.newest_idx = imu_cache.idx_increment(imu_cache.newest_idx);
        imu_cache.imu_queue[newest_idx].time = imuIn->header.stamp.toSec();
        imu_cache.imu_queue[newest_idx].roll = roll;
        imu_cache.imu_queue[newest_idx].pitch = pitch;
    }

    void publish_tf() {
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                  (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odom_mapped_.header.stamp = ros::Time().fromSec(time_odom_);
        odom_mapped_.pose.pose.orientation.x = -geoQuat.y;
        odom_mapped_.pose.pose.orientation.y = -geoQuat.z;
        odom_mapped_.pose.pose.orientation.z = geoQuat.x;
        odom_mapped_.pose.pose.orientation.w = geoQuat.w;
        odom_mapped_.pose.pose.position.x = transformAftMapped[3];
        odom_mapped_.pose.pose.position.y = transformAftMapped[4];
        odom_mapped_.pose.pose.position.z = transformAftMapped[5];
        odom_mapped_.twist.twist.angular.x = transformBefMapped[0];
        odom_mapped_.twist.twist.angular.y = transformBefMapped[1];
        odom_mapped_.twist.twist.angular.z = transformBefMapped[2];
        odom_mapped_.twist.twist.linear.x = transformBefMapped[3];
        odom_mapped_.twist.twist.linear.y = transformBefMapped[4];
        odom_mapped_.twist.twist.linear.z = transformBefMapped[5];
        pub_odom_after_mapped_.publish(odom_mapped_);

        transform_mapped_.stamp_ = ros::Time().fromSec(time_odom_);
        transform_mapped_.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        transform_mapped_.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tf_broadcaster_.sendTransform(transform_mapped_);
    }

    PointPose trans2PointTypePose(float transformIn[]) {
        PointPose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    void publish_key_poses_frames() {
        sensor_msgs::PointCloud2 laser_cloud_temp;

        if (pub_key_poses_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*key_poses_3d_, laser_cloud_temp);
            laser_cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            laser_cloud_temp.header.frame_id = "camera_init";
            pub_key_poses_.publish(laser_cloud_temp);
        }

        if (pub_recent_key_frames_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*surf_map_ds_, laser_cloud_temp);
            laser_cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            laser_cloud_temp.header.frame_id = "camera_init";
            pub_recent_key_frames_.publish(laser_cloud_temp);
        }

        if (pub_registered_cloud_.getNumSubscribers() != 0) {
            pcl::PointCloud<Point>::Ptr cloudOut(new pcl::PointCloud<Point>);
            PointPose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(last_corner_ds_,  &thisPose6D);
            *cloudOut += *transformPointCloud(last_total_surf_, &thisPose6D);
            
            pcl::toROSMsg(*cloudOut, laser_cloud_temp);
            laser_cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            laser_cloud_temp.header.frame_id = "camera_init";
            pub_registered_cloud_.publish(laser_cloud_temp);
        } 
    }

    void visualizeGlobalMapThread() {
        ros::Rate rate(0.2);
        while (ros::ok()) {
            rate.sleep();
            publish_globalmap();
        }
        // save final point cloud
        pcl::io::savePCDFileASCII(fileDirectory+"finalCloud.pcd", *global_map_key_frames_ds_);

        string cornerMapString = "/tmp/cornerMap.pcd";
        string surfaceMapString = "/tmp/surfaceMap.pcd";
        string trajectoryString = "/tmp/trajectory.pcd";

        pcl::PointCloud<Point>::Ptr cornerMapCloud(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr cornerMapCloudDS(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr surfaceMapCloud(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr surfaceMapCloudDS(new pcl::PointCloud<Point>);
        
        for(int i = 0; i < corner_key_frames_.size(); i++) {
            *cornerMapCloud  += *transformPointCloud(corner_key_frames_[i],   &key_poses_6d_->points[i]);
    	    *surfaceMapCloud += *transformPointCloud(surf_key_frames_[i],     &key_poses_6d_->points[i]);
    	    *surfaceMapCloud += *transformPointCloud(recent_key_frames_[i],  &key_poses_6d_->points[i]);
        }

        vg_corner_filter_.setInputCloud(cornerMapCloud);
        vg_corner_filter_.filter(*cornerMapCloudDS);
        vg_surf_filter_.setInputCloud(surfaceMapCloud);
        vg_surf_filter_.filter(*surfaceMapCloudDS);

        pcl::io::savePCDFileASCII(fileDirectory+"cornerMap.pcd", *cornerMapCloudDS);
        pcl::io::savePCDFileASCII(fileDirectory+"surfaceMap.pcd", *surfaceMapCloudDS);
        pcl::io::savePCDFileASCII(fileDirectory+"trajectory.pcd", *key_poses_3d_);
    }

    void publish_globalmap() {
        if (pub_laser_cloud_surround_.getNumSubscribers() == 0)
            return;

        if (key_poses_3d_->points.empty() == true)
            return;

        std::vector<int> closest_indices;
        std::vector<float> closest_square_distances;
	    // search near key frames to visualize
        mutex_.lock();
        kdtree_global_map_->setInputCloud(key_poses_3d_);
        kdtree_global_map_->radiusSearch(current_pose_, globalMapVisualizationSearchRadius, closest_indices, closest_square_distances, 0);
        mutex_.unlock();

        for (int i : closest_indices) {
            global_map_key_poses_->points.push_back(key_poses_3d_->points[i]);
        }
        vg_global_map_key_poses_filter_.setInputCloud(global_map_key_poses_);
        vg_global_map_key_poses_filter_.filter(*global_map_key_poses_ds_);

	    // extract visualized and downsampled key frames
        for (const auto &p : global_map_key_poses_ds_->points) {
			int i = (int)p.intensity;
			*global_map_key_frames_ += *transformPointCloud(corner_key_frames_[i],   &key_poses_6d_->points[i]);
			*global_map_key_frames_ += *transformPointCloud(surf_key_frames_[i],    &key_poses_6d_->points[i]);
			*global_map_key_frames_ += *transformPointCloud(recent_key_frames_[i], &key_poses_6d_->points[i]);
        }
        vg_global_map_key_frames_filter_.setInputCloud(global_map_key_frames_);
        vg_global_map_key_frames_filter_.filter(*global_map_key_frames_ds_);
 
        sensor_msgs::PointCloud2 laser_cloud_temp;
        pcl::toROSMsg(*global_map_key_frames_ds_, laser_cloud_temp);
        laser_cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
        laser_cloud_temp.header.frame_id = "camera_init";
        pub_laser_cloud_surround_.publish(laser_cloud_temp);  

        global_map_key_poses_->clear();
        global_map_key_poses_ds_->clear();
        global_map_key_frames_->clear();
        // global_map_key_frames_ds_->clear();     
    }

    void loopClosureThread() {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(1);
        while (ros::ok()) {
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure() {
        lastest_surf_key_frame_->clear();
        near_history_surf_key_frame_->clear();
        near_history_surf_key_frame_ds_->clear();

        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<int> closest_indices;
        std::vector<float> closest_square_distances;
        kdtree_history_key_poses_->setInputCloud(key_poses_3d_);
        kdtree_history_key_poses_->radiusSearch(current_pose_, historyKeyframeSearchRadius, closest_indices, closest_square_distances, 0);
        
        closest_history_frame_id_ = -1;
        for (int i : closest_indices) {
            if (abs(key_poses_6d_->points[i].time - time_odom_) > 30.0) {
                closest_history_frame_id_ = i;
                break;
            }
        }
        if (closest_history_frame_id_ == -1) {
            return false;
        }
        // save latest key frames
        lastest_frame_id_loop_closure_ = key_poses_3d_->points.size() - 1;
        *lastest_surf_key_frame_ += *transformPointCloud(corner_key_frames_[lastest_frame_id_loop_closure_], &key_poses_6d_->points[lastest_frame_id_loop_closure_]);
        *lastest_surf_key_frame_ += *transformPointCloud(surf_key_frames_[lastest_frame_id_loop_closure_],   &key_poses_6d_->points[lastest_frame_id_loop_closure_]);

        pcl::PointCloud<Point>::Ptr hahaCloud(new pcl::PointCloud<Point>);
        for (const auto &p : lastest_surf_key_frame_->points) {
            if ((int)p.intensity >= 0) {
                hahaCloud->push_back(p);
            }
        }
        lastest_surf_key_frame_.swap(hahaCloud);

	   // save history near key frames
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j) {
            if (closest_history_frame_id_ + j < 0 || closest_history_frame_id_ + j > lastest_frame_id_loop_closure_)
                continue;
            *near_history_surf_key_frame_ += *transformPointCloud(corner_key_frames_[closest_history_frame_id_+j], &key_poses_6d_->points[closest_history_frame_id_+j]);
            *near_history_surf_key_frame_ += *transformPointCloud(surf_key_frames_[closest_history_frame_id_+j],   &key_poses_6d_->points[closest_history_frame_id_+j]);
        }

        vg_history_key_frames_filter_.setInputCloud(near_history_surf_key_frame_);
        vg_history_key_frames_filter_.filter(*near_history_surf_key_frame_ds_);
        // publish history near key frames
        if (pub_history_key_frames_.getNumSubscribers() != 0) {
            sensor_msgs::PointCloud2 laser_cloud_temp;
            pcl::toROSMsg(*near_history_surf_key_frame_ds_, laser_cloud_temp);
            laser_cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            laser_cloud_temp.header.frame_id = "camera_init";
            pub_history_key_frames_.publish(laser_cloud_temp);
        }

        return true;
    }

    void performLoopClosure() {
        static bool is_potential_loop = false;

        if (key_poses_3d_->points.empty() == true)
            return;
        // try to find close key frame if there are any
        if (!is_potential_loop) {
            is_potential_loop = detectLoopClosure(); // find some key frames that is old enough or close enough for loop closure
        }
        if (!is_potential_loop)
            return;

        // reset the flag first no matter icp successes or not
        is_potential_loop = false;
        // ICP Settings
        pcl::IterativeClosestPoint<Point, Point> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);
        // Align clouds
        icp.setInputSource(lastest_surf_key_frame_);
        icp.setInputTarget(near_history_surf_key_frame_ds_);
        pcl::PointCloud<Point>::Ptr unused_result(new pcl::PointCloud<Point>);
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;
        // publish corrected cloud
        if (pub_icp_key_frames_.getNumSubscribers() != 0) {
            pcl::PointCloud<Point>::Ptr closed_cloud(new pcl::PointCloud<Point>);
            pcl::transformPointCloud(*lastest_surf_key_frame_, *closed_cloud, icp.getFinalTransformation());
            sensor_msgs::PointCloud2 laser_cloud_temp;
            pcl::toROSMsg(*closed_cloud, laser_cloud_temp);
            laser_cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            laser_cloud_temp.header.frame_id = "camera_init";
            pub_icp_key_frames_.publish(laser_cloud_temp);
        }   
        /*
        	get pose constraint
        	*/
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
        pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
        Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(key_poses_6d_->points[lastest_frame_id_loop_closure_]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = point_to_gtpose(key_poses_6d_->points[closest_history_frame_id_]);
        gtsam::Vector Vector6(6);
        float noise_score = icp.getFitnessScore();
        Vector6 << noise_score, noise_score, noise_score, noise_score, noise_score, noise_score;
        constraint_noise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
        /* 
        	add constraints
        	*/
        std::unique_lock<std::mutex> lock(mutex_);
        gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(lastest_frame_id_loop_closure_, closest_history_frame_id_, poseFrom.between(poseTo), constraint_noise_));
        isam_->update(gtsam_graph_);
        isam_->update();
        gtsam_graph_.resize(0);

        is_closure_loop_ = true;
    }

    gtsam::Pose3 point_to_gtpose(const PointPose &p) { // camera frame to lidar frame
    	return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(p.yaw), double(p.roll), double(p.pitch)),
                           Point3(double(p.z), double(p.x), double(p.y)));
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointPose thisPoint) { // camera frame to lidar frame
    	return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
    }

    void extract_surrounding_key_frames() {
        if (key_poses_3d_->points.empty() == true)
            return;	
		
    	if (loopClosureEnableFlag == true) {
    	    // only use recent key poses for graph building
                if (recent_corner_key_frames_.size() < surroundingKeyframeSearchNum) { // queue is not full (the beginning of mapping or a loop is just closed)
                    // clear recent key frames queue
                    recent_corner_key_frames_. clear();
                    recent_surf_key_frames_.   clear();
                    recent_outlier_key_frames_.clear();
                    for (auto it = key_poses_3d_->points.rbegin();
                        it != key_poses_3d_->points.rend();
                        ++it) {
                        int i = (int)it->intensity;
                        update_transform_sin_cos(key_poses_6d_->points[i]);
                        // extract surrounding map
                        recent_corner_key_frames_.push_front(transformPointCloud(corner_key_frames_[i]));
                        recent_surf_key_frames_.push_front(transformPointCloud(surf_key_frames_[i]));
                        recent_outlier_key_frames_.push_front(transformPointCloud(recent_key_frames_[i]));
                        if (recent_corner_key_frames_.size() >= surroundingKeyframeSearchNum)
                            break;
                    }
                }else{  // queue is full, pop the oldest key frame and push the latest key frame
                    if (latest_frame_id_ != key_poses_3d_->points.size() - 1) {  // if the robot is not moving, no need to update recent frames
                        recent_corner_key_frames_.pop_front();
                        recent_surf_key_frames_.pop_front();
                        recent_outlier_key_frames_.pop_front();

                        // push latest scan to the end of queue
                        latest_frame_id_ = key_poses_3d_->points.size() - 1;
                        update_transform_sin_cos(key_poses_6d_->points[latest_frame_id_]);
                        recent_corner_key_frames_.push_back(transformPointCloud(corner_key_frames_[latest_frame_id_]));
                        recent_surf_key_frames_.push_back(transformPointCloud(surf_key_frames_[latest_frame_id_]));
                        recent_outlier_key_frames_.push_back(transformPointCloud(recent_key_frames_[latest_frame_id_]));
                    }
                }

                for (int i = 0; i < recent_corner_key_frames_.size(); ++i) {
                    *corner_map_ += *recent_corner_key_frames_[i];
                    *surf_map_ += *recent_surf_key_frames_[i];
                    *surf_map_ += *recent_outlier_key_frames_[i];
                }
    	}else{
            surrounding_key_poses_->clear();
            surrounding_key_poses_ds_->clear();
    	    // extract all the nearby key poses and downsample them
    	    kdtree_surrounding_key_poses_->setInputCloud(key_poses_3d_);
    	    kdtree_surrounding_key_poses_->radiusSearch(current_pose_, (double)surroundingKeyframeSearchRadius, closest_indices_, closest_square_distances_, 0);
    	    for (int i : closest_indices_) {
                surrounding_key_poses_->points.push_back(key_poses_3d_->points[i]);
            }
    	    vg_surrounding_key_poses_filter_.setInputCloud(surrounding_key_poses_);
    	    vg_surrounding_key_poses_filter_.filter(*surrounding_key_poses_ds_);
    	    // delete key frames that are not in surrounding region
            for (int i = 0; i < surrounding_existing_key_poses_id_.size(); ++i) {
                bool existingFlag = false;
                for (const auto &p : surrounding_key_poses_ds_->points.size()) {
                    if (surrounding_existing_key_poses_id_[i] == (int)p.intensity) {
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == false) {
                    surrounding_existing_key_poses_id_.erase(surrounding_existing_key_poses_id_.   begin() + i);
                    surrounding_corner_key_frames_.erase(surrounding_corner_key_frames_. begin() + i);
                    surrounding_surf_key_frames_.erase(surrounding_surf_key_frames_.   begin() + i);
                    surrounding_outlier_key_frames_.erase(surrounding_outlier_key_frames_.begin() + i);
                    --i;
                }
            }
    	    // add new key frames that are not in calculated existing key frames
            for (const auto &p : surrounding_key_poses_ds_->points.size()) {
                bool existingFlag = false;
                for (const auto &s : surrounding_existing_key_poses_id_) {
                    if (s == (int)p.intensity) {
                        existingFlag = true;
                        break;
                    }
                }
                if (!existingFlag) {
                    int i = (int)p.intensity;
                    update_transform_sin_cos(key_poses_6d_->points[i]);
                    surrounding_existing_key_poses_id_.push_back(i);
                    surrounding_corner_key_frames_.push_back(transformPointCloud(corner_key_frames_[i]));
                    surrounding_surf_key_frames_.push_back(transformPointCloud(surf_key_frames_[i]));
                    surrounding_outlier_key_frames_.push_back(transformPointCloud(recent_key_frames_[i]));
                }
            }

            for (int i = 0; i < surrounding_existing_key_poses_id_.size(); ++i) {
                *corner_map_ += *surrounding_corner_key_frames_[i];
                *surf_map_ += *surrounding_surf_key_frames_[i];
                *surf_map_ += *surrounding_outlier_key_frames_[i];
            }
    	}
        // Downsample the surrounding corner key frames (or map)
        vg_corner_filter_.setInputCloud(corner_map_);
        vg_corner_filter_.filter(*corner_map_ds_);

        // Downsample the surrounding surf key frames (or map)
        vg_surf_filter_.setInputCloud(surf_map_);
        vg_surf_filter_.filter(*surf_map_ds_);
    }

    void downsample_current_scan() {
        vg_corner_filter_.setInputCloud(cloud_last_corner_);
        vg_corner_filter_.filter(*last_corner_ds_);

        vg_surf_filter_.setInputCloud(cloud_last_surf_);
        vg_surf_filter_.filter(*last_surf_ds_);

        vg_outlier_filter_.setInputCloud(cloud_last_outlier_);
        vg_outlier_filter_.filter(*last_outlier_ds_);

        *last_total_surf_ += *last_surf_ds_;
        *last_total_surf_ += *last_outlier_ds_;
        vg_surf_filter_.setInputCloud(last_total_surf_);
        vg_surf_filter_.filter(*last_total_surf_ds_);
    }

    void cornerOptimization(int iterCount) {
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < last_corner_ds_->points.size(); i++) {
            const auto &p = laserCloudCornerLastDS->points[i];
            auto p_map = pointAssociateToMap(p);
            kdtree_corner_map_->nearestKSearch(p_map, 5, closest_indices_, closest_square_distances_);

            if (closest_square_distances_[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += corner_map_ds_->points[closest_indices_[j]].x;
                    cy += corner_map_ds_->points[closest_indices_[j]].y;
                    cz += corner_map_ds_->points[closest_indices_[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = corner_map_ds_->points[closest_indices_[j]].x - cx;
                    float ay = corner_map_ds_->points[closest_indices_[j]].y - cy;
                    float az = corner_map_ds_->points[closest_indices_[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    float x0 = p_map.x;
                    float y0 = p_map.y;
                    float z0 = p_map.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                    * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;
                    float s = 1 - 0.9 * fabs(ld2);

                    if (s > 0.1) {
                        cloud_ori_->push_back(p);

                        Point coeff;
                        coeff.x = s * la;
                        coeff.y = s * lb;
                        coeff.z = s * lc;
                        coeff.intensity = s * ld2;

                        coeff_sel_->push_back(coeff);
                    }
                }
            }
        }
    }

    void surfOptimization(int iterCount) {
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < last_total_surf_ds_->points.size(); i++) {
            const auto &p = last_total_surf_ds_->points[i];
            auto p_map = pointAssociateToMap(p); 
            kdtree_surf_map_->nearestKSearch(p_map, 5, closest_indices_, closest_square_distances_);

            if (closest_square_distances_[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0.at<float>(j, 0) = surf_map_ds_->points[closest_indices_[j]].x;
                    matA0.at<float>(j, 1) = surf_map_ds_->points[closest_indices_[j]].y;
                    matA0.at<float>(j, 2) = surf_map_ds_->points[closest_indices_[j]].z;
                }
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * surf_map_ds_->points[closest_indices_[j]].x +
                             pb * surf_map_ds_->points[closest_indices_[j]].y +
                             pc * surf_map_ds_->points[closest_indices_[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * p_map.x + pb * p_map.y + pc * p_map.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(p_map.x * p_map.x
                            + p_map.y * p_map.y + p_map.z * p_map.z));

                    if (s > 0.1) {
                        cloud_ori_->push_back(p);

                        Point coeff;
                        coeff.x = s * pa;
                        coeff.y = s * pb;
                        coeff.z = s * pc;
                        coeff.intensity = s * pd2;
                        coeff_sel_->push_back(coeff);
                    }
                }
            }
        }
    }

    bool LMOptimization(int iterCount) {
        float srx = std::sin(transformTobeMapped[0]);
        float crx = std::cos(transformTobeMapped[0]);
        float sry = std::sin(transformTobeMapped[1]);
        float cry = std::cos(transformTobeMapped[1]);
        float srz = std::sin(transformTobeMapped[2]);
        float crz = std::cos(transformTobeMapped[2]);

        int laserCloudSelNum = cloud_ori_->points.size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        for (int i = 0; i < laserCloudSelNum; i++) {
            const auto &p = cloud_ori_->points[i];
            const auto &coeff = coeff_sel_->points[i];

            float arx = (crx*sry*srz*p.x + crx*crz*sry*p.y - srx*sry*p.z) * coeff.x
                      + (-srx*srz*p.x - crz*srx*p.y - crx*p.z) * coeff.y
                      + (crx*cry*srz*p.x + crx*cry*crz*p.y - cry*srx*p.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*p.x 
                      + (sry*srz + cry*crz*srx)*p.y + crx*cry*p.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*p.x 
                      + (cry*srz - crz*srx*sry)*p.y - crx*sry*p.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*p.x + (-cry*crz-srx*sry*srz)*p.y)*coeff.x
                      + (crx*crz*.x - crx*srz*p.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*p.x + (crz*sry-cry*srx*srz)*p.y)*coeff.z;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
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
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
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

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }

    void scan2MapOptimization() {
        if (corner_map_ds_->points.size() > 10 && surf_map_ds_->points.size() > 100) {
            kdtree_corner_map_->setInputCloud(corner_map_ds_);
            kdtree_surf_map_->setInputCloud(surf_map_ds_);

            for (int i = 0; i < 10; i++) {

                cloud_ori_->clear();
                coeff_sel_->clear();

                cornerOptimization(i);
                surfOptimization(i);

                if (LMOptimization(i) == true)
                    break;              
            }

            transform_update();
        }
    }

    void save_key_frames_factor() {
        current_pose_.x = transformAftMapped[3];
        current_pose_.y = transformAftMapped[4];
        current_pose_.z = transformAftMapped[5];

        bool saveThisKeyFrame = true;
        if (sqrt((prev_pose_.x-current_pose_.x)*(prev_pose_.x-current_pose_.x)
                +(prev_pose_.y-current_pose_.y)*(prev_pose_.y-current_pose_.y)
                +(prev_pose_.z-current_pose_.z)*(prev_pose_.z-current_pose_.z)) < 0.3) {
            saveThisKeyFrame = false;
        }

        

        if (saveThisKeyFrame == false && !key_poses_3d_->points.empty())
        	return;

        prev_pose_ = current_pose_;
        /**
         * update grsam graph
         */
        if (key_poses_3d_->points.empty()) {
            gtsam_graph_.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                       		 Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), prior_noise_));
            initial_estimate_.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                                  Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
            	transform_last_[i] = transformTobeMapped[i];
        }
        else{
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transform_last_[2], transform_last_[0], transform_last_[1]),
                                                Point3(transform_last_[5], transform_last_[3], transform_last_[4]));
            gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));
            gtsam_graph_.add(BetweenFactor<Pose3>(key_poses_3d_->points.size()-1, key_poses_3d_->points.size(), poseFrom.between(poseTo), odometry_noise_));
            initial_estimate_.insert(key_poses_3d_->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                                     		   Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));
        }
        /**
         * update isam_
         */
        isam_->update(gtsam_graph_, initial_estimate_);
        isam_->update();
        
        gtsam_graph_.resize(0);
        initial_estimate_.clear();

        /**
         * save key poses
         */
        Point thisPose3D;
        PointPose thisPose6D;
        Pose3 latestEstimate;

        current_estimate_ = isam_->calculateEstimate();
        latestEstimate = current_estimate_.at<Pose3>(current_estimate_.size()-1);

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = key_poses_3d_->points.size(); // this can be used as index
        key_poses_3d_->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw   = latestEstimate.rotation().roll(); // in camera frame
        thisPose6D.time = time_odom_;
        key_poses_6d_->push_back(thisPose6D);
        /**
         * save updated transform
         */
        if (key_poses_3d_->points.size() > 1) {
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();

            for (int i = 0; i < 6; ++i) {
            	transform_last_[i] = transformAftMapped[i];
            	transformTobeMapped[i] = transformAftMapped[i];
            }
        }

        pcl::PointCloud<Point>::Ptr thisCornerKeyFrame(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr thisSurfKeyFrame(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<Point>);

        pcl::copy_point_cloud(*last_corner_ds_,  *thisCornerKeyFrame);
        pcl::copy_point_cloud(*last_surf_ds_,    *thisSurfKeyFrame);
        pcl::copy_point_cloud(*last_outlier_ds_, *thisOutlierKeyFrame);

        corner_key_frames_.push_back(thisCornerKeyFrame);
        surf_key_frames_.push_back(thisSurfKeyFrame);
        recent_key_frames_.push_back(thisOutlierKeyFrame);
    }

    void correct_poses() {
    	if (is_closure_loop_ == true) {
            recent_corner_key_frames_.clear();
            recent_surf_key_frames_.clear();
            recent_outlier_key_frames_.clear();

            // update key poses
            for (int i = 0; i < current_estimate_.size(); ++i) {
            key_poses_3d_->points[i].x = current_estimate_.at<Pose3>(i).translation().y();
            key_poses_3d_->points[i].y = current_estimate_.at<Pose3>(i).translation().z();
            key_poses_3d_->points[i].z = current_estimate_.at<Pose3>(i).translation().x();

            key_poses_6d_->points[i].x = key_poses_3d_->points[i].x;
            key_poses_6d_->points[i].y = key_poses_3d_->points[i].y;
            key_poses_6d_->points[i].z = key_poses_3d_->points[i].z;
            key_poses_6d_->points[i].roll  = current_estimate_.at<Pose3>(i).rotation().pitch();
            key_poses_6d_->points[i].pitch = current_estimate_.at<Pose3>(i).rotation().yaw();
            key_poses_6d_->points[i].yaw   = current_estimate_.at<Pose3>(i).rotation().roll();
            }

            is_closure_loop_ = false;
        }
    }

    void clear_cloud() {
        corner_map_->clear();
        surf_map_->clear();  
        corner_map_ds_->clear();
        surf_map_ds_->clear();   
    }

    void run() {

        if (has_get_corner_last_ && std::abs(time_corner_last_ - time_odom_) < 0.005 &&
            has_get_surf_last_ && std::abs(time_surf_last_ - time_odom_) < 0.005 &&
            has_get_outlier_last_ && std::abs(time_outlier_last_ - time_odom_) < 0.005 &&
            has_get_laser_odom_)
        {
            has_get_corner_last_ = false;
            has_get_surf_last_ = false;
            has_get_outlier_last_ = false;
            has_get_laser_odom_ = false;

            std::unique_lock<std::mutex> lock(mutex_);

            if (time_odom_ - time_last_process_ >= mappingProcessInterval) {
                time_last_process_ = time_odom_;

                transformAssociateToMap();

                extract_surrounding_key_frames();

                downsample_current_scan();

                scan2MapOptimization();

                save_key_frames_factor();

                correct_poses();

                publish_tf();

                publish_key_poses_frames();

                clear_cloud();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    mapOptimization MO;

    std::thread loop_thread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualize_thread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();
        MO.run();
        rate.sleep();
    }

    loop_thread.join();
    visualize_thread.join();

    return 0;
}