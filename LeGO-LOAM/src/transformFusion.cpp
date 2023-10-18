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

#include "utility.h"
#include "lego_math.h"

class TransformFusion{
private:
    ros::NodeHandle nh_;

    ros::Publisher pub_laser_odom_;
    ros::Subscriber sub_laser_odom_;
    ros::Subscriber sub_odom_mapped_;

    std_msgs::Header current_header_;
    nav_msgs::Odometry laser_odom_;

    tf::StampedTransform transform_laser_odom_;
    tf::StampedTransform transform_map_2_camera_;
    tf::StampedTransform transform_camera_2_baselink_;
    tf::TransformBroadcaster tf_broadcaster_;

    float transform_from_first_laser_frame_[6];
    float transformIncre[6];
    float transformMapped[6];
    float transformBefMapped[6];
    float transformAftMapped[6];

public:
    TransformFusion() {
        pub_laser_odom_ = nh_.advertise<nav_msgs::Odometry>("/integrated_to_init", 5);
        sub_laser_odom_ = nh_.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &TransformFusion::laser_odom_handler, this);
        sub_odom_mapped_ = nh_.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 5, &TransformFusion::odom_mapped_handler, this);

        laser_odom_.header.frame_id = "camera_init";
        laser_odom_.child_frame_id = "camera";

        transform_laser_odom_.frame_id_ = "camera_init";
        transform_laser_odom_.child_frame_id_ = "camera";

        transform_map_2_camera_.frame_id_ = "map";
        transform_map_2_camera_.child_frame_id_ = "camera_init";

        transform_camera_2_baselink_.frame_id_ = "camera";
        transform_camera_2_baselink_.child_frame_id_ = "base_link";

        for (int i = 0; i < 6; ++i)
        {
            transform_from_first_laser_frame_[i] = 0;
            transformIncre[i] = 0;
            transformMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }
    }

    void transformAssociateToMap()
    {
        float sbcx = std::sin(transform_from_first_laser_frame_[0]);
        float cbcx = std::cos(transform_from_first_laser_frame_[0]);
        float sbcy = std::sin(transform_from_first_laser_frame_[1]);
        float cbcy = std::cos(transform_from_first_laser_frame_[1]);
        float sbcz = std::sin(transform_from_first_laser_frame_[2]);
        float cbcz = std::cos(transform_from_first_laser_frame_[2]);

        auto r0 = rotate_by_yxz(transformBefMapped[3] - transform_from_first_laser_frame_[3],
                                transformBefMapped[4] - transform_from_first_laser_frame_[4],
                                transformBefMapped[5] - transform_from_first_laser_frame_[5],
                                cbcx, -sbcx,
                                cbcy, -sbcy,
                                cbcz, -sbcz);

        transformIncre[3] = r0[0];
        transformIncre[4] = r0[1];
        transformIncre[5] = r0[2];

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
        transformMapped[0] = -asin(srx);

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
        transformMapped[1] = std::atan2(srycrx / std::cos(transformMapped[0]), 
                                   crycrx / std::cos(transformMapped[0]));
        
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
        transformMapped[2] = std::atan2(srzcrx / std::cos(transformMapped[0]), 
                                   crzcrx / std::cos(transformMapped[0]));

        auto r1 = rotate_by_zxy(transformIncre[3],
                                transformIncre[4],
                                transformIncre[5],
                                std::cos(transformMapped[0]),
                                std::sin(transformMapped[0]),
                                std::cos(transformMapped[1]),
                                std::sin(transformMapped[1]),
                                std::cos(transformMapped[2]),
                                std::sin(transformMapped[2]));

        transformMapped[3] = transformAftMapped[3] - r1[0];
        transformMapped[4] = transformAftMapped[4] - r1[1];
        transformMapped[5] = transformAftMapped[5] - r1[2];
    }

    void laser_odom_handler(const nav_msgs::Odometry::ConstPtr &laser_odometry_)
    {
        current_header_ = laser_odometry_->header;

        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laser_odometry_->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);

        transform_from_first_laser_frame_[0] = -pitch;
        transform_from_first_laser_frame_[1] = -yaw;
        transform_from_first_laser_frame_[2] = roll;

        transform_from_first_laser_frame_[3] = laser_odometry_->pose.pose.position.x;
        transform_from_first_laser_frame_[4] = laser_odometry_->pose.pose.position.y;
        transform_from_first_laser_frame_[5] = laser_odometry_->pose.pose.position.z;

        transformAssociateToMap();

        geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                  (transformMapped[2], -transformMapped[0], -transformMapped[1]);

        laser_odom_.header.stamp = laser_odometry_->header.stamp;
        laser_odom_.pose.pose.orientation.x = -geoQuat.y;
        laser_odom_.pose.pose.orientation.y = -geoQuat.z;
        laser_odom_.pose.pose.orientation.z = geoQuat.x;
        laser_odom_.pose.pose.orientation.w = geoQuat.w;
        laser_odom_.pose.pose.position.x = transformMapped[3];
        laser_odom_.pose.pose.position.y = transformMapped[4];
        laser_odom_.pose.pose.position.z = transformMapped[5];
        pub_laser_odom_.publish(laser_odom_);

        transform_laser_odom_.stamp_ = laser_odometry_->header.stamp;
        transform_laser_odom_.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        transform_laser_odom_.setOrigin(tf::Vector3(transformMapped[3], transformMapped[4], transformMapped[5]));
        tf_broadcaster_.sendTransform(transform_laser_odom_);
    }

    void odom_mapped_handler(const nav_msgs::Odometry::ConstPtr &odom_mapped)
    {
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = odom_mapped->pose.pose.orientation;
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);

        transformAftMapped[0] = -pitch;
        transformAftMapped[1] = -yaw;
        transformAftMapped[2] = roll;

        transformAftMapped[3] = odom_mapped->pose.pose.position.x;
        transformAftMapped[4] = odom_mapped->pose.pose.position.y;
        transformAftMapped[5] = odom_mapped->pose.pose.position.z;

        transformBefMapped[0] = odom_mapped->twist.twist.angular.x;
        transformBefMapped[1] = odom_mapped->twist.twist.angular.y;
        transformBefMapped[2] = odom_mapped->twist.twist.angular.z;

        transformBefMapped[3] = odom_mapped->twist.twist.linear.x;
        transformBefMapped[4] = odom_mapped->twist.twist.linear.y;
        transformBefMapped[5] = odom_mapped->twist.twist.linear.z;
    }
};


int main(int argc, const char **argv)
{
    ros::init(argc, argv, "lego_loam");
    
    TransformFusion tf;

    ROS_INFO("\033[1;32m---->\033[0m Transform Fusion Started.");

    ros::spin();

    return 0;
}
