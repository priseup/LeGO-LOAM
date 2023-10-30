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
#include <array>

class ImageProjection{
public:
    ImageProjection();

    void cloud_handler(const sensor_msgs::PointCloud2ConstPtr& laser_cloud);

private:
    // Convert ros message to pcl point cloud
    void copy_point_cloud(const sensor_msgs::PointCloud2ConstPtr &laser_cloud);

    // Start and end angle of a scan
    void calculate_orientation();

    // Range image projection
    void project_point_cloud();

    // Mark ground points
    void extract_ground();

    // Point cloud segmentation
    void extract_segmentation();

    // Publish all clouds
    void publish_cloud();

    void allocate_memory();

    // Reset parameters for next iteration
    void reset_parameters();

    void bfs_cluster(int row, int col);

    int point_row(const Point &p, int idx) const;
    int point_column(const Point &p) const;

private:
    struct QueueElement
    {
        int row;
        int col;
    };

    struct Queue
    {
        // int cluster_size = 0;
        int start = 0;
        int end = 0;

        struct QueueElement elements[N_SCAN*Horizon_SCAN];
    };
    enum class PointLabel {
        valid,
        invalid,
        outlier,
        ground,
        segmentation
    };

private:

    int file_idx_ = 0;
    ros::NodeHandle nh_;

    ros::Subscriber sub_laser_cloud_;
    
    ros::Publisher pub_projected_cloud_;
    ros::Publisher pub_projected_cloud_with_range_;

    ros::Publisher pub_pure_ground_cloud_;
    ros::Publisher pub_ground_segment_cloud_;
    ros::Publisher pub_pure_segmented_cloud_;
    ros::Publisher pub_segmented_cloud_info_;
    ros::Publisher pub_outlier_cloud_;

    pcl::PointCloud<Point>::Ptr laser_cloud_input_;
    std::vector<int> laser_cloud_ring_;

    pcl::PointCloud<Point>::Ptr projected_laser_cloud_; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<Point>::Ptr projected_cloud_with_range_; // same as projected_laser_cloud_, but with intensity  range

    pcl::PointCloud<Point>::Ptr projected_pure_ground_cloud_;
    pcl::PointCloud<Point>::Ptr projected_ground_segment_cloud_;
    pcl::PointCloud<Point>::Ptr projected_pure_segmented_cloud_;
    pcl::PointCloud<Point>::Ptr projected_outlier_cloud_;

    std::vector<PointLabel> point_label_;
    std::vector<int> point_cluster_id_;

    int segment_id_ = 1;

    Point init_point_value_; // fill in projected_laser_cloud_ at each iteration

    cv::Mat projected_cloud_range_; // range matrix for range image

    cloud_msgs::cloud_info segmented_cloud_msg_; // info of segmented cloud
    std_msgs::Header cloud_header_;

    std::array<std::pair<int8_t, int8_t>, 4> neighbors_; // neighbor iterator for segmentaiton process
};
