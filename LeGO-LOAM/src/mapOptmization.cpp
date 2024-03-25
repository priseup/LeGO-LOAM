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
#include <array>
#include <memory>
#include <unordered_set>
#include <thread>
#include <functional>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <std_srvs/Empty.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include "utility.h"
#include "lego_math.h"
#include "ceres_wrapper.h"
#include "eigen_wrapper.h"

// Mapping Params
const double surroundKeyframeSearchRadius = 50.0; // key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
const int   surroundKeyframeSearchNum = 50; // submap size (when loop closure enabled)
// history key frames (history submap for loop closure)
const double historyKeyframeSearchRadius = 7.0; // key frame that is within n meters from current pose will be considerd for loop closure
const int   historyKeyframeSearchNum = 25; // 2n+1 number of hostory key frames will be used into a submap for loop closure
const float historyKeyframeFitnessScore = 0.3; // the smaller the better alignment

const float globalMapVisualizationSearchRadius = 500.0; // key frames with in n meters will be visualized

const double mappingProcessInterval = 0.3;

class mapOptimization{
private:
private:
    gtsam::NonlinearFactorGraph fator_graph_;
    gtsam::Values initial_estimate_;
    gtsam::Values optimized_estimate_;
    gtsam::Values current_estimate_;
    std::unique_ptr<gtsam::ISAM2> isam_;

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odometry_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr constraint_noise_;

    ros::NodeHandle nh_;

    ros::Publisher pub_surround_;
    ros::Publisher pub_odom_after_mapped_;
    ros::Publisher pub_keyposes_;

    ros::Publisher pub_history_keyframes_;
    ros::Publisher pub_icp_keyframes_;
    ros::Publisher pub_recent_keyframes_;
    ros::Publisher pub_registered_cloud_;

    ros::Subscriber sub_corner_;
    ros::Subscriber sub_surf_;
    ros::Subscriber sub_outlier_;
    ros::Subscriber sub_laser_odom_;

    ros::ServiceServer srv_save_map_;

    nav_msgs::Odometry odom_mapped_;

    std::vector<pcl::PointCloud<Point>::Ptr> corner_keyframes_;
    std::vector<pcl::PointCloud<Point>::Ptr> surf_keyframes_;
    std::vector<pcl::PointCloud<Point>::Ptr> outlier_keyframes_;

    std::deque<pcl::PointCloud<Point>::Ptr> recent_corner_keyframes_;
    std::deque<pcl::PointCloud<Point>::Ptr> recent_surf_keyframes_;
    std::deque<pcl::PointCloud<Point>::Ptr> recent_outlier_keyframes_;
    int latest_frame_id_ = 0;

    std::vector<int> surround_exist_keyposes_id_;
    std::deque<pcl::PointCloud<Point>::Ptr> surround_corner_keyframes_;
    std::deque<pcl::PointCloud<Point>::Ptr> surround_surf_keyframes_;
    std::deque<pcl::PointCloud<Point>::Ptr> surround_outlier_keyframes_;
    
    Point prev_pose_;
    Point current_pose_;

    pcl::PointCloud<Point>::Ptr keyposes_3d_;
    pcl::PointCloud<PointPose>::Ptr keyposes_6d_;

    pcl::PointCloud<Point>::Ptr surround_keyposes_;
    pcl::PointCloud<Point>::Ptr surround_keyposes_ds_;

    pcl::PointCloud<Point>::Ptr corner_;
    pcl::PointCloud<Point>::Ptr surf_;
    pcl::PointCloud<Point>::Ptr outlier_;
    pcl::PointCloud<Point>::Ptr surf_outlier_;

    pcl::PointCloud<Point>::Ptr corner_map_;
    pcl::PointCloud<Point>::Ptr corner_map_ds_;
    pcl::PointCloud<Point>::Ptr surf_map_;
    pcl::PointCloud<Point>::Ptr surf_map_ds_;

    pcl::KdTreeFLANN<Point>::Ptr kd_corner_map_;
    pcl::KdTreeFLANN<Point>::Ptr kd_surf_map_;

    pcl::KdTreeFLANN<Point>::Ptr kd_surround_keyposes_;
    pcl::KdTreeFLANN<Point>::Ptr kd_history_keyposes_;
    
    pcl::PointCloud<Point>::Ptr near_history_surf_keyframe_;

    pcl::PointCloud<Point>::Ptr lastest_corner_keyframe_;
    pcl::PointCloud<Point>::Ptr lastest_surf_keyframe_;
    pcl::PointCloud<Point>::Ptr lastest_surf_keyframe_ds_;

    pcl::KdTreeFLANN<Point>::Ptr kd_global_map_;
    pcl::PointCloud<Point>::Ptr global_map_keyposes_;
    pcl::PointCloud<Point>::Ptr global_map_keyframes_;

    pcl::VoxelGrid<Point> ds_corner_;
    pcl::VoxelGrid<Point> ds_surf_;
    pcl::VoxelGrid<Point> ds_outlier_;
    pcl::VoxelGrid<Point> ds_history_keyframes_; // for histor key frames of loop closure
    pcl::VoxelGrid<Point> ds_map_keyposes_; // for global map visualization
    pcl::VoxelGrid<Point> ds_map_keyframes_; // for global map visualization

    double time_corner_ = 0;
    double time_surf_ = 0;
    double time_outlier_ = 0;
    double time_odom_ = 0;

    bool has_get_corner_ = false;
    bool has_get_surf_ = false;
    bool has_get_outlier_ = false;
    bool has_get_laser_odom_ = false;

    // double pose_params_[7] = {0, 0. 0, 1, 0, 0, 0}; // x, y, z, w, tx, ty, tz
    // Eigen::Map<Eigen::Quaterniond> q_w_curr_(pose_params_);
    // Eigen::Map<Eigen::Vector3d>  t_w_curr_(pose_params_ + 4);

    double params_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    Eigen::Quaterniond q_map_odom_;
    Eigen::Vector3d t_map_odom_;

    Eigen::Quaterniond q_odom_curr_;
    Eigen::Vector3d t_odom_curr_;

    Eigen::Quaterniond q_map_laser_;
    Eigen::Vector3d t_map_laser_;

    std::mutex mutex_;
    Eigen::Matrix4d correction_;

    double time_process_ = -1;

    int closest_history_frame_id_ = -1;
    int lastest_frame_id_loop_closure_;

    bool is_closure_loop_ = false;

public:
    mapOptimization(): nh_("~")
    {
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam_ = std::make_unique<gtsam::ISAM2>(gtsam::ISAM2(parameters));

        pub_keyposes_ = nh_.advertise<sensor_msgs::PointCloud2>("/trajectory", 2);
        pub_surround_ = nh_.advertise<sensor_msgs::PointCloud2>("/surround_cloud", 2);
        pub_odom_after_mapped_ = nh_.advertise<nav_msgs::Odometry>("/odom_after_mapped", 5);

        sub_corner_ = nh_.subscribe<sensor_msgs::PointCloud2>("/corner_last", 2, &mapOptimization::corner_handler, this);
        sub_surf_ = nh_.subscribe<sensor_msgs::PointCloud2>("/surf_last", 2, &mapOptimization::surf_handler, this);
        sub_outlier_ = nh_.subscribe<sensor_msgs::PointCloud2>("/outlier_last", 2, &mapOptimization::outlier_handler, this);
        sub_laser_odom_ = nh_.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laser_odom_handler, this);

        pub_history_keyframes_ = nh_.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pub_icp_keyframes_ = nh_.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pub_recent_keyframes_ = nh_.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
        pub_registered_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/registered_cloud", 2);

        srv_save_map_ = nh_.advertiseService("save_map", &mapOptimization::save_map_handler, this);

        ds_corner_.setLeafSize(0.2, 0.2, 0.2);
        ds_surf_.setLeafSize(0.4, 0.4, 0.4);
        ds_outlier_.setLeafSize(0.4, 0.4, 0.4);

        ds_history_keyframes_.setLeafSize(0.4, 0.4, 0.4); // for histor key frames of loop closure

        ds_map_keyposes_.setLeafSize(1.0, 1.0, 1.0); // for global map visualization
        ds_map_keyframes_.setLeafSize(0.4, 0.4, 0.4); // for global map visualization

        odom_mapped_.header.frame_id = "map";
        odom_mapped_.child_frame_id = "rslidar";

        q_map_odom_.setIdentity();
        t_map_odom_.setZero();

        q_odom_curr_.setIdentity();
        t_odom_curr_.setZero();

        q_map_laser_.setIdentity();
        t_map_laser_.setZero();

        correction_.setIdentity();
        allocate_memory();
    }

    void allocate_memory() {
        keyposes_3d_.reset(new pcl::PointCloud<Point>);
        keyposes_6d_.reset(new pcl::PointCloud<PointPose>);

        kd_surround_keyposes_.reset(new pcl::KdTreeFLANN<Point>);
        kd_history_keyposes_.reset(new pcl::KdTreeFLANN<Point>);

        surround_keyposes_.reset(new pcl::PointCloud<Point>);

        corner_.reset(new pcl::PointCloud<Point>); // corner feature set from odoOptimization
        surf_.reset(new pcl::PointCloud<Point>); // surf feature set from odoOptimization
        outlier_.reset(new pcl::PointCloud<Point>); // corner feature set from odoOptimization
        surf_outlier_.reset(new pcl::PointCloud<Point>); // surf feature set from odoOptimization

        corner_map_.reset(new pcl::PointCloud<Point>);
        surf_map_.reset(new pcl::PointCloud<Point>);
        corner_map_ds_.reset(new pcl::PointCloud<Point>);
        surf_map_ds_.reset(new pcl::PointCloud<Point>);

        kd_corner_map_.reset(new pcl::KdTreeFLANN<Point>);
        kd_surf_map_.reset(new pcl::KdTreeFLANN<Point>);
        
        near_history_surf_keyframe_.reset(new pcl::PointCloud<Point>);

        lastest_corner_keyframe_.reset(new pcl::PointCloud<Point>);
        lastest_surf_keyframe_.reset(new pcl::PointCloud<Point>);
        lastest_surf_keyframe_ds_.reset(new pcl::PointCloud<Point>);

        kd_global_map_.reset(new pcl::KdTreeFLANN<Point>);
        global_map_keyposes_.reset(new pcl::PointCloud<Point>);
        global_map_keyframes_.reset(new pcl::PointCloud<Point>);

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        prior_noise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
        odometry_noise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
    }

    void transform_to_map()
    {
        q_map_laser_ = q_map_odom_ * q_odom_curr_;
        t_map_laser_ = q_map_odom_ * t_odom_curr_ + t_map_odom_;
    }

    void transform_update()
    {
        q_map_laser_ = Eigen::AngleAxisd(params_[5], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(params_[4], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(params_[3], Eigen::Vector3d::UnitX());
        t_map_laser_[0] = params_[0];
        t_map_laser_[1] = params_[1];
        t_map_laser_[2] = params_[2];

        q_map_odom_ = q_map_laser_ * q_odom_curr_.inverse();
        t_map_odom_ = t_map_laser_ - q_map_odom_ * t_odom_curr_;
    }

    Point laser_point_to_map(const Point &p)
    {
        auto point_w = q_map_laser_ * to_vector(p) + t_map_laser_;

        Point po;
        po.x = point_w.x();
        po.y = point_w.y();
        po.z = point_w.z();
        po.intensity = p.intensity;

        return po;
    }

    pcl::PointCloud<Point>::Ptr transformPointCloud(pcl::PointCloud<Point>::Ptr cloud_in) {
	// !!! DO NOT use pcl for point cloud transformation, results are not accurate
        // Reason: unknown
        // PointPose pose;
        pcl::PointCloud<Point>::Ptr cloud_out(new pcl::PointCloud<Point>);
        cloud_out->resize(cloud_in->points.size());

/*
        for (int i = 0; i < cloud_in->points.size(); ++i) {
            const auto &p = cloud_in->points[i];
            auto &po = cloud_out->points[i];

            auto r = rotate_by_zxy(p.x, p.y, p.z,
                                ctRoll, stRoll,
                                ctPitch, stPitch,
                                ctYaw, stYaw);

            po.x = r[0] + tInX;
            po.y = r[1] + tInY;
            po.z = r[2] + tInZ;
            po.intensity = p.intensity;
        }
*/
        return cloud_out;
    }

    pcl::PointCloud<Point>::Ptr transformPointCloud(pcl::PointCloud<Point>::Ptr cloud_in, const PointPose &pose) {
        Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
        transform.block<3, 3>(0, 0) = q_map_laser_.toRotationMatrix().cast<float>();
        transform(0, 3) = pose.x;
        transform(1, 3) = pose.y;
        transform(2, 3) = pose.z;

        pcl::PointCloud<Point>::Ptr cloud_out(new pcl::PointCloud<Point>);
        pcl::transformPointCloud(*cloud_in, *cloud_out, transform);
        
        return cloud_out;
    }

    void outlier_handler(const sensor_msgs::PointCloud2ConstPtr& msg) {
        time_outlier_ = msg->header.stamp.toSec();
        pcl::fromROSMsg(*msg, *outlier_);
        has_get_outlier_ = true;
    }

    void corner_handler(const sensor_msgs::PointCloud2ConstPtr& msg) {
        time_corner_ = msg->header.stamp.toSec();
        pcl::fromROSMsg(*msg, *corner_);
        has_get_corner_ = true;
    }

    void surf_handler(const sensor_msgs::PointCloud2ConstPtr& msg) {
        time_surf_ = msg->header.stamp.toSec();
        pcl::fromROSMsg(*msg, *surf_);
        has_get_surf_ = true;
    }

    void laser_odom_handler(const nav_msgs::Odometry::ConstPtr &odom) {
        time_odom_ = odom->header.stamp.toSec();
        has_get_laser_odom_ = true;

        t_odom_curr_(0) = odom->pose.pose.position.x;
        t_odom_curr_(1) = odom->pose.pose.position.y;
        t_odom_curr_(2) = odom->pose.pose.position.z;

        q_odom_curr_.w() = odom->pose.pose.orientation.w;
        q_odom_curr_.x() = odom->pose.pose.orientation.x;
        q_odom_curr_.y() = odom->pose.pose.orientation.y;
        q_odom_curr_.z() = odom->pose.pose.orientation.z;

        transform_to_map();

/*
        tf::Transform tf_map_odom;
        tf_map_odom.setOrigin(tf::Vector3(t_map_odom_.x(), t_map_odom_.y(), t_map_odom_.z()));
        tf_map_odom.setRotation(tf::Quaternion(q_map_odom_.x(), q_map_odom_.y(), q_map_odom_.z(), q_map_odom_.w()));
        tf_broadcaster_.sendTransform(tf::StampedTransform(tf_map_odom, time_odom_, "map", "odom"));
*/
    }

    bool save_map_handler(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res) {
        pcl::PointCloud<Point>::Ptr map_trajectory(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr map_corner(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr map_surf(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr map_outlier(new pcl::PointCloud<Point>);

        for (int i = 0; i < keyposes_3d_->size(); ++i) {
            map_trajectory->push_back(keyposes_3d_->points[i]);
            map_trajectory->back().intensity = i;

            *map_corner += *transformPointCloud(corner_keyframes_[i], keyposes_6d_->points[i]);
            *map_surf += *transformPointCloud(surf_keyframes_[i], keyposes_6d_->points[i]);
            *map_outlier += *transformPointCloud(outlier_keyframes_[i], keyposes_6d_->points[i]);
        }

        pcl::io::savePCDFile("/home/pqf/my_lego/trajectory.pcd", *map_trajectory);
        pcl::io::savePCDFile("/home/pqf/my_lego/corner.pcd", *map_corner);
        pcl::io::savePCDFile("/home/pqf/my_lego/surface.pcd", *map_surf);
        pcl::io::savePCDFile("/home/pqf/my_lego/outlier.pcd", *map_outlier);

        return true;
    }

    void publish_tf() {
        odom_mapped_.header.stamp = ros::Time().fromSec(time_odom_);
        odom_mapped_.pose.pose.orientation.x = q_map_laser_.x();
        odom_mapped_.pose.pose.orientation.y = q_map_laser_.y();
        odom_mapped_.pose.pose.orientation.z = q_map_laser_.z();
        odom_mapped_.pose.pose.orientation.w = q_map_laser_.w();
        odom_mapped_.pose.pose.position.x = t_map_laser_[0];
        odom_mapped_.pose.pose.position.y = t_map_laser_[1];
        odom_mapped_.pose.pose.position.z = t_map_laser_[2];
        pub_odom_after_mapped_.publish(odom_mapped_);

        tf::Transform tf_map_odom;
        tf_map_odom.setOrigin(tf::Vector3(t_map_odom_.x(), t_map_odom_.y(), t_map_odom_.z()));
        tf_map_odom.setRotation(tf::Quaternion(q_map_odom_.x(), q_map_odom_.y(), q_map_odom_.z(), q_map_odom_.w()));

        tf::TransformBroadcaster tf_broadcaster;
        tf_broadcaster.sendTransform(tf::StampedTransform(tf_map_odom, ros::Time().fromSec(time_odom_), "map", "odom"));
    }

    void publish_keyposes_frames() {
        sensor_msgs::PointCloud2 cloud_temp;

        if (pub_keyposes_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*keyposes_3d_, cloud_temp);
            cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            cloud_temp.header.frame_id = "map";
            pub_keyposes_.publish(cloud_temp);
        }

        if (pub_recent_keyframes_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*surf_map_ds_, cloud_temp);
            cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            cloud_temp.header.frame_id = "map";
            pub_recent_keyframes_.publish(cloud_temp);
        }

        if (pub_registered_cloud_.getNumSubscribers() != 0) {
            pcl::PointCloud<Point>::Ptr cloud_out(new pcl::PointCloud<Point>);
            // PointPose thisPose6D = transform_to_pose(transformTobeMapped);
            PointPose thisPose6D;
            thisPose6D.x = t_map_laser_[0];
            thisPose6D.y = t_map_laser_[1];
            thisPose6D.roll = q_map_laser_.x();
            thisPose6D.pitch = q_map_laser_.y();
            thisPose6D.yaw = q_map_laser_.z();
            *cloud_out += *transformPointCloud(corner_,  thisPose6D);
            *cloud_out += *transformPointCloud(surf_outlier_, thisPose6D);
            
            pcl::toROSMsg(*cloud_out, cloud_temp);
            cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            cloud_temp.header.frame_id = "map";
            pub_registered_cloud_.publish(cloud_temp);
        } 
    }

    void visual_global_map() {
        ros::Rate rate(0.2);
        while (ros::ok()) {
            rate.sleep();
            publish_globalmap();
        }
        // save final point cloud
        pcl::io::savePCDFileASCII(fileDirectory+"finalCloud.pcd", *global_map_keyframes_);

        std::string cornerMapString = "/tmp/cornerMap.pcd";
        std::string surfaceMapString = "/tmp/surfaceMap.pcd";
        std::string trajectoryString = "/tmp/trajectory.pcd";

        pcl::PointCloud<Point>::Ptr cornerMapCloud(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr cornerMapCloudDS(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr surfaceMapCloud(new pcl::PointCloud<Point>);
        pcl::PointCloud<Point>::Ptr surfaceMapCloudDS(new pcl::PointCloud<Point>);
        
        for(int i = 0; i < corner_keyframes_.size(); i++) {
          *cornerMapCloud  += *transformPointCloud(corner_keyframes_[i], keyposes_6d_->points[i]);
    	    *surfaceMapCloud += *transformPointCloud(surf_keyframes_[i], keyposes_6d_->points[i]);
    	    *surfaceMapCloud += *transformPointCloud(outlier_keyframes_[i], keyposes_6d_->points[i]);
        }

        ds_corner_.setInputCloud(cornerMapCloud);
        ds_corner_.filter(*cornerMapCloudDS);
        ds_surf_.setInputCloud(surfaceMapCloud);
        ds_surf_.filter(*surfaceMapCloudDS);

        pcl::io::savePCDFileASCII(fileDirectory+"cornerMap.pcd", *cornerMapCloudDS);
        pcl::io::savePCDFileASCII(fileDirectory+"surfaceMap.pcd", *surfaceMapCloudDS);
        pcl::io::savePCDFileASCII(fileDirectory+"trajectory.pcd", *keyposes_3d_);
    }

    void publish_globalmap() {
        if (pub_surround_.getNumSubscribers() == 0)
            return;

        if (keyposes_3d_->points.empty())
            return;

        std::vector<int> closest_indices;
        std::vector<float> closest_square_distances;
	    // search near key frames to visualize
        mutex_.lock();
        kd_global_map_->setInputCloud(keyposes_3d_);
        kd_global_map_->radiusSearch(current_pose_, globalMapVisualizationSearchRadius, closest_indices, closest_square_distances);
        mutex_.unlock();

        for (int i : closest_indices) {
            global_map_keyposes_->push_back(keyposes_3d_->points[i]);
        }
        ds_map_keyposes_.setInputCloud(global_map_keyposes_);
        ds_map_keyposes_.filter(*global_map_keyposes_);

	    // extract visualized and downsampled key frames
        for (const auto &p : global_map_keyposes_->points) {
            int i = (int)p.intensity;
            *global_map_keyframes_ += *transformPointCloud(corner_keyframes_[i], keyposes_6d_->points[i]);
            *global_map_keyframes_ += *transformPointCloud(surf_keyframes_[i], keyposes_6d_->points[i]);
            *global_map_keyframes_ += *transformPointCloud(outlier_keyframes_[i], keyposes_6d_->points[i]);
        }
        ds_map_keyframes_.setInputCloud(global_map_keyframes_);
        ds_map_keyframes_.filter(*global_map_keyframes_);
 
        sensor_msgs::PointCloud2 cloud_temp;
        pcl::toROSMsg(*global_map_keyframes_, cloud_temp);
        cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
        cloud_temp.header.frame_id = "map";
        pub_surround_.publish(cloud_temp);  

        global_map_keyposes_->clear();
        // global_map_keyframes_->clear();
    }

    void loop_closure_thread() {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(1);
        while (ros::ok()) {
            rate.sleep();
            perform_loop_closure();
        }
    }

    bool detect_loop_closure() {
        lastest_surf_keyframe_->clear();
        near_history_surf_keyframe_->clear();

        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<int> closest_indices;
        std::vector<float> closest_square_distances;
        kd_history_keyposes_->setInputCloud(keyposes_3d_);
        kd_history_keyposes_->radiusSearch(current_pose_, historyKeyframeSearchRadius, closest_indices, closest_square_distances);
        
        closest_history_frame_id_ = -1;
        for (int i : closest_indices) {
            if (std::abs(keyposes_6d_->points[i].time - time_odom_) > 30.0) {
                closest_history_frame_id_ = i;
                break;
            }
        }
        if (closest_history_frame_id_ == -1) {
            return false;
        }
        // save latest key frames
        lastest_frame_id_loop_closure_ = keyposes_3d_->points.size() - 1;
        *lastest_surf_keyframe_ += *transformPointCloud(corner_keyframes_[lastest_frame_id_loop_closure_], keyposes_6d_->points[lastest_frame_id_loop_closure_]);
        *lastest_surf_keyframe_ += *transformPointCloud(surf_keyframes_[lastest_frame_id_loop_closure_], keyposes_6d_->points[lastest_frame_id_loop_closure_]);

        pcl::PointCloud<Point>::Ptr hahaCloud(new pcl::PointCloud<Point>);
        for (const auto &p : lastest_surf_keyframe_->points) {
            if ((int)p.intensity >= 0) {
                hahaCloud->push_back(p);
            }
        }
        lastest_surf_keyframe_.swap(hahaCloud);

	   // save history near key frames
        for (int i = -historyKeyframeSearchNum; i <= historyKeyframeSearchNum; ++i) {
            if (closest_history_frame_id_ + i < 0 || closest_history_frame_id_ + i > lastest_frame_id_loop_closure_)
                continue;
            *near_history_surf_keyframe_ += *transformPointCloud(corner_keyframes_[closest_history_frame_id_+i], keyposes_6d_->points[closest_history_frame_id_+i]);
            *near_history_surf_keyframe_ += *transformPointCloud(surf_keyframes_[closest_history_frame_id_+i], keyposes_6d_->points[closest_history_frame_id_+i]);
        }

        ds_history_keyframes_.setInputCloud(near_history_surf_keyframe_);
        ds_history_keyframes_.filter(*near_history_surf_keyframe_);

        // publish history near key frames
        if (pub_history_keyframes_.getNumSubscribers() != 0) {
            sensor_msgs::PointCloud2 cloud_temp;
            pcl::toROSMsg(*near_history_surf_keyframe_, cloud_temp);
            cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            cloud_temp.header.frame_id = "map";
            pub_history_keyframes_.publish(cloud_temp);
        }

        return true;
    }

    void perform_loop_closure() {
        static bool is_potential_loop = false;

        if (keyposes_3d_->points.empty())
            return;
        // try to find close key frame if there are any
        if (!is_potential_loop) {
            is_potential_loop = detect_loop_closure(); // find some key frames that is old enough or close enough for loop closure
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
        icp.setInputSource(lastest_surf_keyframe_);
        icp.setInputTarget(near_history_surf_keyframe_);
        pcl::PointCloud<Point>::Ptr unused_result(new pcl::PointCloud<Point>);
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;
        // publish corrected cloud
        if (pub_icp_keyframes_.getNumSubscribers() != 0) {
            pcl::PointCloud<Point>::Ptr closed_cloud(new pcl::PointCloud<Point>);
            pcl::transformPointCloud(*lastest_surf_keyframe_, *closed_cloud, icp.getFinalTransformation());
            sensor_msgs::PointCloud2 cloud_temp;
            pcl::toROSMsg(*closed_cloud, cloud_temp);
            cloud_temp.header.stamp = ros::Time().fromSec(time_odom_);
            cloud_temp.header.frame_id = "map";
            pub_icp_keyframes_.publish(cloud_temp);
        }   
        /*
        	get pose constraint
        	*/
        Eigen::Matrix4f correction_frame = icp.getFinalTransformation();
        Eigen::Matrix4f t_correct = correction_frame;
        Eigen::Quaternionf r_correct(t_correct.block<3, 3>(0, 0));
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::Quaternion(r_correct.w(), r_correct.x(), r_correct.y(), r_correct.z()), gtsam::Point3(t_correct(0, 3), t_correct(1, 3), t_correct(2, 3)));
        gtsam::Pose3 poseTo = point_to_gtpose(keyposes_6d_->points[closest_history_frame_id_]);
        gtsam::Vector Vector6(6);
        float noise_score = icp.getFitnessScore();
        Vector6 << noise_score, noise_score, noise_score, noise_score, noise_score, noise_score;
        constraint_noise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
        /* 
        	add constraints
        	*/
        std::unique_lock<std::mutex> lock(mutex_);
        fator_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(lastest_frame_id_loop_closure_, closest_history_frame_id_, poseFrom.between(poseTo), constraint_noise_));
        isam_->update(fator_graph_);
        isam_->update();
        fator_graph_.resize(0);

        correction_ = correction_frame.cast<double>();

        is_closure_loop_ = true;
    }

    gtsam::Pose3 point_to_gtpose(const PointPose &p) {
    	return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(p.roll), double(p.pitch), double(p.yaw)),
                           gtsam::Point3(double(p.x), double(p.y), double(p.z)));
    }

    void extract_surround_keyframes() {
        if (keyposes_3d_->points.empty())
            return;	
		
    	if (loopClosureEnableFlag) {
          // only use recent key poses for graph building
          if (recent_corner_keyframes_.size() < surroundKeyframeSearchNum) { // queue is not full (the beginning of mapping or a loop is just closed)
              // clear recent key frames queue
              recent_corner_keyframes_.clear();
              recent_surf_keyframes_.clear();
              recent_outlier_keyframes_.clear();
              for (auto it = keyposes_3d_->points.rbegin();
                  it != keyposes_3d_->points.rend();
                  ++it) {
                  int i = (int)it->intensity; // intensity, key frame index in corner/surf/outlier_keyframes
                  // extract surround map
                  recent_corner_keyframes_.push_front(transformPointCloud(corner_keyframes_[i], keyposes_6d_->points[i]));
                  recent_surf_keyframes_.push_front(transformPointCloud(surf_keyframes_[i], keyposes_6d_->points[i]));
                  recent_outlier_keyframes_.push_front(transformPointCloud(outlier_keyframes_[i], keyposes_6d_->points[i]));
                  if (recent_corner_keyframes_.size() >= surroundKeyframeSearchNum)
                      break;
              }
          }else{  // queue is full, pop the oldest key frame and push the latest key frame
              if (latest_frame_id_ != keyposes_3d_->points.size() - 1) {  // if the robot is not moving, no need to update recent frames
                  recent_corner_keyframes_.pop_front();
                  recent_surf_keyframes_.pop_front();
                  recent_outlier_keyframes_.pop_front();

                  // push latest scan to the end of queue
                  latest_frame_id_ = keyposes_3d_->points.size() - 1;
                  recent_corner_keyframes_.push_back(transformPointCloud(corner_keyframes_[latest_frame_id_], keyposes_6d_->points[latest_frame_id_]));
                  recent_surf_keyframes_.push_back(transformPointCloud(surf_keyframes_[latest_frame_id_], keyposes_6d_->points[latest_frame_id_]));
                  recent_outlier_keyframes_.push_back(transformPointCloud(outlier_keyframes_[latest_frame_id_], keyposes_6d_->points[latest_frame_id_]));
              }
          }

          for (int i = 0; i < recent_corner_keyframes_.size(); ++i) {
              *corner_map_ += *recent_corner_keyframes_[i];
              *surf_map_ += *recent_surf_keyframes_[i];
              *surf_map_ += *recent_outlier_keyframes_[i];
          }
    	}else{
          surround_keyposes_->clear();
        // extract all the nearby key poses and downsample them
          std::vector<int> closest_indices;
          std::vector<float> closest_square_distances;
    	    kd_surround_keyposes_->setInputCloud(keyposes_3d_);
    	    kd_surround_keyposes_->radiusSearch(current_pose_, surroundKeyframeSearchRadius, closest_indices, closest_square_distances);
    	    for (int i : closest_indices) {
              surround_keyposes_->push_back(keyposes_3d_->points[i]);
          }
          static pcl::VoxelGrid<Point> ds_keyposes; // for surround key poses of scan-to-map optimization
          ds_keyposes.setLeafSize(1.0, 1.0, 1.0); // for surround key poses of scan-to-map optimization
    	    ds_keyposes.setInputCloud(surround_keyposes_);
    	    ds_keyposes.filter(*surround_keyposes_ds_);

    	    // delete key frames that are not in surround region
          std::set<int> delete_indices;
          for (int i = 0; i < surround_exist_keyposes_id_.size(); i++) {
             bool is_exist = false;
             for (auto &p : surround_keyposes_ds_->points) {
                 if ((int)p.intensity == surround_exist_keyposes_id_[i]) {
                     is_exist = true;
                     break;
                 }
             }
             if (!is_exist) {
                 delete_indices.insert(i);
             }
          }

          for (auto it = delete_indices.rbegin(); it != delete_indices.rend(); ++it)
          {
              surround_exist_keyposes_id_.erase(surround_exist_keyposes_id_.begin() + *it);
              surround_corner_keyframes_.erase(surround_corner_keyframes_.begin() + *it);
              surround_surf_keyframes_.erase(surround_surf_keyframes_.begin() + *it);
              surround_outlier_keyframes_.erase(surround_outlier_keyframes_.begin() + *it);
          }
          delete_indices.clear();

          // add new key frames that are not in calculated existing key frames
          for (const auto &p : surround_keyposes_ds_->points) {
              int idx = (int)p.intensity;
              if (std::count(surround_exist_keyposes_id_.begin(), surround_exist_keyposes_id_.end(), idx) == 0) {
                  surround_exist_keyposes_id_.push_back(idx);
                  surround_corner_keyframes_.push_back(transformPointCloud(corner_keyframes_[idx], keyposes_6d_->points[idx]));
                  surround_surf_keyframes_.push_back(transformPointCloud(surf_keyframes_[idx], keyposes_6d_->points[idx]));
                  surround_outlier_keyframes_.push_back(transformPointCloud(outlier_keyframes_[idx], keyposes_6d_->points[idx]));
              }
          }

          for (int i = 0; i < surround_exist_keyposes_id_.size(); ++i) {
              *corner_map_ += *surround_corner_keyframes_[i];
              *surf_map_ += *surround_surf_keyframes_[i];
              *surf_map_ += *surround_outlier_keyframes_[i];
          }
    	}
      // Downsample the surround corner key frames (or map)
      ds_corner_.setInputCloud(corner_map_);
      ds_corner_.filter(*corner_map_ds_);

      // Downsample the surround surf key frames (or map)
      ds_surf_.setInputCloud(surf_map_);
      ds_surf_.filter(*surf_map_ds_);
    }

    void downsample_current_scan() {
        ds_corner_.setInputCloud(corner_);
        ds_corner_.filter(*corner_);

        ds_surf_.setInputCloud(surf_);
        ds_surf_.filter(*surf_);

        ds_outlier_.setInputCloud(outlier_);
        ds_outlier_.filter(*outlier_);

        *surf_outlier_ += *surf_;
        *surf_outlier_ += *outlier_;
        ds_surf_.setInputCloud(surf_outlier_);
        ds_surf_.filter(*surf_outlier_);
    }

    void corner_optimization(ceres::Problem& problem, ceres::LossFunction *loss_function) {
        for (const auto &p : corner_->points) {
            auto p_map = laser_point_to_map(p);

            std::vector<int> closest_indices;
            std::vector<float> closest_square_distances;
            kd_corner_map_->nearestKSearch(p_map, 5, closest_indices, closest_square_distances);

            if (closest_square_distances.back() < 1.0) {
                std::vector<Eigen::Vector3d> nearCorners;
                Eigen::Vector3d center(0., 0., 0.);
                for (int i : closest_indices)
                {
                  Eigen::Vector3d tmp = to_vector(corner_map_ds_->points[i]);
                  center = center + tmp;
                  nearCorners.push_back(tmp);
                }
                center = center / 5.0;

                Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                for (int i = 0; i < 5; i++)
                {
                  Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[i] - center;
                  covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                // if is indeed line feature
                // note Eigen library sort eigenvalues in increasing order
                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                Eigen::Vector3d cp = to_vector(p);
                if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                {
                  Eigen::Vector3d point_on_line = center;
                  Eigen::Vector3d lpj = 0.1 * unit_direction + point_on_line;
                  Eigen::Vector3d lpl = -0.1 * unit_direction + point_on_line;
                  problem.AddResidualBlock(new LidarEdgeCostFunction(cp, lpj, lpl),
                                           loss_function, params_);
                  // ++corner_correspondace;
                }
            }
        }
    }

    void surf_optimization(ceres::Problem &problem, ceres::LossFunction *loss_function) {
        for (const auto &p : surf_outlier_->points) {
            auto p_map = laser_point_to_map(p); 
            std::vector<int> closest_indices;
            std::vector<float> closest_square_distances;
            kd_surf_map_->nearestKSearch(p_map, 5, closest_indices, closest_square_distances);

            if (closest_square_distances.back() < 1.0) {
                Eigen::Matrix<double, 5, 3> matA0;
                Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
                for (int i = 0; i < 5; i++) {
                    int idx = closest_indices[i];
                    matA0(i, 0) = surf_map_ds_->points[idx].x;
                    matA0(i, 1) = surf_map_ds_->points[idx].y;
                    matA0(i, 2) = surf_map_ds_->points[idx].z;
                }
                // find the norm of plane
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double negative_OA_dot_norm = 1 / norm.norm();
                norm.normalize();

                // Here n(pa, pb, pc) is unit norm of plane
                bool planeValid = true;
                for (int i : closest_indices) {
                  // if OX * n > 0.2, then plane is not fit well
                  if (fabs(norm.dot(to_vector(surf_map_ds_->points[i]))
                           + negative_OA_dot_norm) > 0.2)
                  {
                    planeValid = false;
                    ROS_WARN_ONCE("plane is not fit well");
                    break;
                  }
                }
                if (planeValid)
                {
                  Eigen::Vector3d cp = to_vector(p);
                  problem.AddResidualBlock(new LidarPlaneCostFunction(cp, norm, negative_OA_dot_norm),
                                           loss_function, params_);
                  // TODO: 先解决 corner 数量过少的问题，少了十倍
                  // ++surf_correnspondance;
                }
            }
        }
    }

    void scan2map() {
        if (corner_map_ds_->points.size() > 10 && surf_map_ds_->points.size() > 100) {
            kd_corner_map_->setInputCloud(corner_map_ds_);
            kd_surf_map_->setInputCloud(surf_map_ds_);

            for (int i = 0; i < 5; i++) {
                ceres::Problem problem;
                problem.AddParameterBlock(params_, 6);
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

                corner_optimization(problem, loss_function);
                surf_optimization(problem, loss_function);

                solve_problem(problem);
          }
        }
    }

    void save_keyframes_factor() {
        current_pose_.x = t_map_laser_[0];
        current_pose_.y = t_map_laser_[1];
        current_pose_.z = t_map_laser_[2];

        if (!keyposes_3d_->points.empty() && distance(prev_pose_, current_pose_) < 0.3) {
        	return;
        }

        prev_pose_ = current_pose_;
        /**
         * update gtsam graph
         */
        if (keyposes_3d_->points.empty()) {
            gtsam::Pose3 gpose(gtsam::Rot3::Quaternion(q_map_laser_.w(), q_map_laser_.x(), q_map_laser_.y(), q_map_laser_.z()),
                                gtsam::Point3(t_map_laser_.x(), t_map_laser_.y(), t_map_laser_.z()));
            fator_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, gpose, prior_noise_));
            initial_estimate_.insert(0, gpose);
        } else{
            auto &pre_pose = keyposes_6d_->points.back();
            gtsam::Pose3 poseFrom(gtsam::Rot3::RzRyRx(pre_pose.roll, pre_pose.pitch, pre_pose.yaw),
                                gtsam::Point3(pre_pose.x, pre_pose.y, pre_pose.z));
            gtsam::Pose3 poseTo(gtsam::Rot3::Quaternion(q_map_laser_.w(), q_map_laser_.x(), q_map_laser_.y(), q_map_laser_.z()),
                                gtsam::Point3(t_map_laser_.x(), t_map_laser_.y(), t_map_laser_.z()));
            fator_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(keyposes_3d_->points.size()-1, keyposes_3d_->points.size(), poseFrom.between(poseTo), odometry_noise_));
            initial_estimate_.insert(keyposes_3d_->points.size(), poseTo);
        }
        /**
         * update isam_
         */
        isam_->update(fator_graph_, initial_estimate_);
        isam_->update();
        
        fator_graph_.resize(0);
        initial_estimate_.clear();

        /**
         * save key poses
         */
        current_estimate_ = isam_->calculateEstimate();
        gtsam::Pose3 latestEstimate = current_estimate_.at<gtsam::Pose3>(current_estimate_.size() - 1);

        Point pose_3d;
        pose_3d.x = latestEstimate.translation().y();
        pose_3d.y = latestEstimate.translation().z();
        pose_3d.z = latestEstimate.translation().x();
        pose_3d.intensity = keyposes_3d_->size(); // this can be used as index
        keyposes_3d_->push_back(pose_3d);

        PointPose pose_6d;
        pose_6d.x = pose_3d.x;
        pose_6d.y = pose_3d.y;
        pose_6d.z = pose_3d.z;
        pose_6d.intensity = pose_3d.intensity; // this can be used as index
        pose_6d.roll  = latestEstimate.rotation().roll();
        pose_6d.pitch = latestEstimate.rotation().pitch();
        pose_6d.yaw   = latestEstimate.rotation().yaw(); 
        pose_6d.time = time_odom_;
        keyposes_6d_->push_back(pose_6d);
        /**
         * save updated transform
         */
        if (keyposes_3d_->points.size() > 1) {
            params_[0] = pose_6d.x;
            params_[1] = pose_6d.y;
            params_[2] = pose_6d.z;
            params_[3] = pose_6d.roll;
            params_[4] = pose_6d.pitch;
            params_[5] = pose_6d.yaw;

        }

        corner_keyframes_.push_back(corner_->makeShared());
        surf_keyframes_.push_back(surf_->makeShared());
        outlier_keyframes_.push_back(outlier_->makeShared());
    }

    void correct_poses() {
    	if (is_closure_loop_) {
            recent_corner_keyframes_.clear();
            recent_surf_keyframes_.clear();
            recent_outlier_keyframes_.clear();

            // update key poses
            for (int i = 0; i < current_estimate_.size(); ++i) {
                const auto &v = current_estimate_.at<gtsam::Pose3>(i);

                keyposes_3d_->points[i].x = v.translation().y();
                keyposes_3d_->points[i].y = v.translation().z();
                keyposes_3d_->points[i].z = v.translation().x();

                keyposes_6d_->points[i].x = keyposes_3d_->points[i].x;
                keyposes_6d_->points[i].y = keyposes_3d_->points[i].y;
                keyposes_6d_->points[i].z = keyposes_3d_->points[i].z;
                keyposes_6d_->points[i].roll = v.rotation().roll();
                keyposes_6d_->points[i].pitch = v.rotation().pitch();
                keyposes_6d_->points[i].yaw = v.rotation().yaw();
            }

            q_map_odom_ = correction_.block<3, 3>(0, 0) * q_map_odom_.toRotationMatrix();
            t_map_odom_ = correction_.block<3, 3>(0, 0) * t_map_odom_.matrix() + correction_.block<3, 1>(0, 3);
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
        if (has_get_corner_ && std::abs(time_corner_ - time_odom_) < 0.005 &&
            has_get_surf_ && std::abs(time_surf_ - time_odom_) < 0.005 &&
            has_get_outlier_ && std::abs(time_outlier_ - time_odom_) < 0.005 &&
            has_get_laser_odom_)
        {
            has_get_corner_ = false;
            has_get_surf_ = false;
            has_get_outlier_ = false;
            has_get_laser_odom_ = false;

            std::unique_lock<std::mutex> lock(mutex_);

            if (time_odom_ - time_process_ >= mappingProcessInterval) {
                time_process_ = time_odom_;

                transform_to_map();

                extract_surround_keyframes();

                downsample_current_scan();

                scan2map();

                save_keyframes_factor();

                correct_poses();

                publish_tf();

                publish_keyposes_frames();

                clear_cloud();
            }
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    mapOptimization mo;

    std::thread loop_thread(&mapOptimization::loop_closure_thread, &mo);
    std::thread visualize_thread(&mapOptimization::visual_global_map, &mo);

    ros::Rate rate(100);
    while (ros::ok())
    {
        ros::spinOnce();
        mo.run();
        rate.sleep();
    }

    loop_thread.join();
    visualize_thread.join();

    return 0;
}
