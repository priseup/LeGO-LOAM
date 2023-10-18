#include "utility.h"
#include "imageProjection.h"

static int index_in_project_cloud(int row, int col) {
    return row * Horizon_SCAN + col;
}

ImageProjection::ImageProjection(): nh_("~") {
    sub_laser_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloud_handler, this);

    pub_projected_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_projected_with_row", 1);
    pub_projected_cloud_with_range_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_projected_with_range", 1);

    pub_outlier_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/outlier_cloud", 1);
    pub_pure_ground_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/pure_ground_cloud", 1);
    pub_pure_segmented_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/pure_segmented_cluster_cloud", 1);
    pub_ground_segment_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/ground_with_segmented_cloud", 1);
    pub_segmented_cloud_info_ = nh_.advertise<cloud_msgs::cloud_info>("/ground_with_segmented_cloud_info", 1);

    init_point_value_.x = std::numeric_limits<float>::quiet_NaN();
    init_point_value_.y = std::numeric_limits<float>::quiet_NaN();
    init_point_value_.z = std::numeric_limits<float>::quiet_NaN();
    init_point_value_.intensity = -1;

    neighbors_[0] = std::make_pair(-1, 0);
    neighbors_[1] = std::make_pair(1, 0);
    neighbors_[2] = std::make_pair(0, 1);
    neighbors_[3] = std::make_pair(0, -1);

    allocate_memory();
    reset_parameters();
}

void ImageProjection::allocate_memory() {
    const int points_num = N_SCAN*Horizon_SCAN;

    laser_cloud_input_->reset(new pcl::PointCloud<Point>);

    projected_laser_cloud_->reset(new pcl::PointCloud<Point>);
    projected_laser_cloud_->assign(init_point_value_, points_num);

    projected_cloud_with_range_->reset(new pcl::PointCloud<Point>);
    projected_cloud_with_range_->assign(init_point_value_, points_num);

    projected_pure_ground_cloud_->reset(new pcl::PointCloud<Point>);
    projected_ground_segment_cloud_->reset(new pcl::PointCloud<Point>);
    projected_pure_segmented_cloud_->reset(new pcl::PointCloud<Point>);
    projected_outlier_cloud_->reset(new pcl::PointCloud<Point>);

    segmented_cloud_msg_.ring_index_start.assign(N_SCAN, 0);
    segmented_cloud_msg_.ring_index_end.assign(N_SCAN, 0);

    segmented_cloud_msg_.ground_segment_flag.assign(points_num, false);
    segmented_cloud_msg_.ground_segment_cloud_column.assign(points_num, 0);
    segmented_cloud_msg_.ground_segment_cloud_range.assign(points_num, 0);

    point_label_.assign(points_num, ImageProjection::PointLabel::invalid);
}

void ImageProjection::reset_parameters() {
    laser_cloud_input_->clear();
    projected_pure_ground_cloud_->clear();
    projected_pure_segmented_cloud_->clear();
    projected_ground_segment_cloud_->clear();
    projected_outlier_cloud_->clear();

    projected_cloud_range_ = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    projected_cloud_ground_flag_ = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    projected_cloud_label_ = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
}

~ImageProjection() {
}

void ImageProjection::copy_point_cloud(const sensor_msgs::PointCloud2ConstPtr& laser_cloud) {
    cloud_header_ = laser_cloud->header;
    // cloud_header_.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line

    pcl::fromROSMsg(*laser_cloud, *laser_cloud_input_);

    // removeNaNFromPointCloud::
    //      If the data is dense, we don't need to check for NaN
    //      Simply copy the data
    // so, if isDense == true, do nothing???
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laser_cloud_input_, *laser_cloud_input_, indices);

    pcl::PointCloud<PointXYZIR>::Ptr cloud_input_with_ring(new pcl::PointCloud<PointXYZIR>);
    pcl::fromROSMsg(*laser_cloud, *cloud_input_with_ring);
    if (useCloudRing == true) {
        laser_cloud_ring_.resize(laser_cloud_input_->points.size());
        for (int i = 0; i < indices.size(); i++)
        {
            const auto &p = cloud_input_with_ring->points[indices[i]];
            laser_cloud_ring_[i] = p.ring;
        }
    }
}

void ImageProjection::cloud_handler(const sensor_msgs::PointCloud2ConstPtr& laser_cloud) {
    // 1. Convert ros message to pcl point cloud
    copy_point_cloud(laser_cloud);

    // 2. Start and end angle of a scan
    calculate_orientation();

    // 3. Range image projection
    project_point_cloud();

    // 4. Mark ground points
    extract_ground();

    // 5. Point cloud segmentation
    extract_segmentation();

    // 6. Publish all clouds
    publish_cloud();

    // 7. Reset parameters for next iteration
    reset_parameters();
}

void ImageProjection::calculate_orientation() {
    // start and end orientation of this cloud
    segmented_cloud_msg_.orientation_start = -std::atan2(laser_cloud_input_->points[0].y, laser_cloud_input_->points[0].x);
    segmented_cloud_msg_.orientation_end   = -std::atan2(laser_cloud_input_->points[laser_cloud_input_->points.size() - 1].y,
                                                    laser_cloud_input_->points[laser_cloud_input_->points.size() - 1].x) + 2 * M_PI;

    if (segmented_cloud_msg_.orientation_end - segmented_cloud_msg_.orientation_start > 3 * M_PI) {
        segmented_cloud_msg_.orientation_end -= 2 * M_PI;
    } else if (segmented_cloud_msg_.orientation_end - segmented_cloud_msg_.orientation_start < M_PI) {
        segmented_cloud_msg_.orientation_end += 2 * M_PI;
    }

    segmented_cloud_msg_.orientation_diff = segmented_cloud_msg_.orientation_end - segmented_cloud_msg_.orientation_start;
}

int ImageProjection::point_row(const Point &p, int idx) const {
    if (useCloudRing == true)
        return laser_cloud_ring_[idx];

    float vertical_angle = rad2deg(std::atan2(p.z, std::sqrt(p.x * p.x + p.y * p.y)));
    return (vertical_angle + ang_bottom) / laser_resolution_vertical;
}

int ImageProjection::point_column(const Point &p) const {
    float horizon_angle = rad2deg(std::atan2(p.x, p.y));
    int column = -round((horizon_angle - 90.0) / laser_resolution_horizon) + Horizon_SCAN / 2;
    if (column >= Horizon_SCAN)
        column -= Horizon_SCAN;
    return column;
}

// mark point_label_ from invalid to valid
void ImageProjection::project_point_cloud() {
    for (size_t i = 0; i < laser_cloud_input_->points.size(); ++i) {
        const auto &point = laser_cloud_input_->points[i];

        int row = point_row(point, i);
        if (row < 0 || row >= N_SCAN)
        {
            continue;
        }

        int column = point_column(point);
        if (column < 0 || column >= Horizon_SCAN)
        {
            continue;
        }

        float range = laser_range(p);
        if (range < sensorMinimumRange)
        {
            continue;
        }
        projected_cloud_range_.at<float>(row, column) = range;

        int index = index_in_project_cloud(row, column);
        point_label_[index] = ImageProjection::PointLabel::valid;

        projected_laser_cloud_->points[index] = point;
        // intensity will add point_time as a new value in featureAssociation::adjust_distortion()
        // intensity will be row in featureAssociation::find_corresponding_corner/surf_features()
        projected_laser_cloud_->points[index].intensity = row; //  + column / 10000.f;

        projected_cloud_with_range_->points[index] = point;
        projected_cloud_with_range_->points[index].intensity = range;
    }
}

// mark point_label_ from valid to ground
void ImageProjection::extract_ground() {
    for (int j = 0; j < Horizon_SCAN; ++j) {
        for (int i = 0; i < groundScanInd; ++i) {
            int current_idx = index_in_project_cloud(i, j);
            int uppper_idx = index_in_project_cloud(i+1, j);

            if (point_label_[current_idx] == ImageProjection::PointLabel::invalid
                || point_label_[upper_idx] == ImageProjection::PointLabel::invalid) {
                    continue;
                }
                
            float diff_x = projected_laser_cloud_->points[upper_idx].x - projected_laser_cloud_->points[current_idx].x;
            float diff_y = projected_laser_cloud_->points[upper_idx].y - projected_laser_cloud_->points[current_idx].y;
            float diff_z = projected_laser_cloud_->points[upper_idx].z - projected_laser_cloud_->points[current_idx].z;

            float angle = rad2deg(std::atan2(diff_z, sqrt(diff_x*diff_x + diff_y*diff_y)));

            if (abs(angle - sensorMountAngle) <= 10) {
                point_label_[current_idx] = ImageProjection::PointLabel::ground;
                point_label_[upper_idx] = ImageProjection::PointLabel::ground;
            }
        }
    }

    if (pub_pure_ground_cloud_.getNumSubscribers() > 0) {
        for (int i = 0; i <= groundScanInd; ++i) {
            for (int j = 0; j < Horizon_SCAN; ++j) {
                int idx = index_in_project_cloud(i, j);
                if (point_label_[idx] == ImageProjection::PointLabel::ground)
                    projected_pure_ground_cloud_->push_back(projected_laser_cloud_->points[idx]);
            }
        }
    }
}

// mark point_label_ from valid to segmentation and outlier
void ImageProjection::extract_segmentation() {
    for (int i = 0; i < N_SCAN; ++i) {
        for (int j = 0; j < Horizon_SCAN; ++j) {
            int idx = index_in_project_cloud(i, j);
            if (point_label_[idx] == ImageProjection::PointLabel::valid)
                bfs_cluster(i, j);
        }
    }

    int size_of_segment_cloud = 0;
    for (int i = 0; i < N_SCAN; ++i) {
        segmented_cloud_msg_.ring_index_start[i] = size_of_segment_cloud - 1 + 5;
        for (int j = 0; j < Horizon_SCAN; ++j) {
            int idx = index_in_project_cloud(i, j);

            if (point_label_[idx] == ImageProjection::PointLabel::invalid) {
                continue;
            } else if (point_label_[idx] == ImageProjection::PointLabel::outlier) {
                if (i > groundScanInd && j % 5 == 0) {
                    projected_outlier_cloud_->push_back(projected_laser_cloud_->points[idx]);
                }
            } else {
                if (point_label_[idx] == ImageProjection::PointLabel::ground) {
                    if ((j > 5) && (j % 5 != 0) && (j < Horizon_SCAN-5))
                        continue;
                } else {    // PointLabel::segmentation
                    // pure segmented cloud for visualization
                    if (pub_pure_segmented_cloud_.getNumSubscribers() > 0) {
                        projected_pure_segmented_cloud_->push_back(projected_laser_cloud_->points[idx]);
                        projected_pure_segmented_cloud_->points.back().intensity = segmentation_cluster_id_[idx];
                    }
                }

                projected_ground_segment_cloud_->push_back(projected_laser_cloud_->points[idx]);

                // mark ground points so they will not be considered as edge features later
                segmented_cloud_msg_.ground_segment_flag[size_of_segment_cloud] = point_label_[idx] == ImageProjection::PointLabel::ground;
                // mark the points' column index for marking occlusion later
                segmented_cloud_msg_.ground_segment_cloud_column[size_of_segment_cloud] = j;
                // save range info
                segmented_cloud_msg_.ground_segment_cloud_range[size_of_segment_cloud] = projected_cloud_range_.at<float>(i,j);

                // size of seg cloud
                ++size_of_segment_cloud;
            }
        }
        segmented_cloud_msg_.ring_index_end[i] = size_of_segment_cloud - 1 - 5;
    }
}

// use std::queue std::vector std::deque will slow the program down greatly
void ImageProjection::bfs_cluster(int row, int col) {
    struct QueueElement
    {
        int row;
        int col;
    };

    struct Queue
    {
        int cluster_size = 0;
        int start = 0;
        int end = 0;

        struct QueueElement elements[N_SCAN*Horizon_SCAN]
    };

    static int segment_id = 1;
    static bool cross_scan_flag[N_SCAN] = {false};
    static struct Queue queue;
    queue.elements[0].row = row;
    queue.elements[0].col = col;
    queue.start = 0;
    queue.end = 1;
    queue.cluster_size = queue.end;

    while (queue.end - queue.start > 0) {
        int current_row = queue.element[start].row;
        int current_col = queue.element[start].col;
        ++queue.start;   // pop front

        int idx = index_in_project_cloud(current_row, current_col);
        point_label_[idx] = ImageProjection::PointLabel::segmentation;
        segmentation_cluster_id_[idx] = segment_id;

        for (auto &n : neighbors_) {
            int neibor_row = current_row + n.first;
            int neibor_col = current_col + n.second;

            // index should be within the boundary
            if (neibor_row < 0 || neibor_row >= N_SCAN)
                continue;
            // at range image margin (left or right side)
            if (neibor_col < 0)
                neibor_col = Horizon_SCAN - 1;
            if (neibor_col >= Horizon_SCAN)
                neibor_col = 0;

            // prevent infinite loop (caused by put already examined point back)
            if (segmentation_cluster_id_.find(idx) != segmentation_cluster.end())
                continue;

            auto &&distance = std::minmax(projected_cloud_range_.at<float>(current_row, current_col), 
                            projected_cloud_range_.at<float>(neibor_row, neibor_col));

            float alpha = 0.f;
            if (e.first == 0) // row
                alpha = segmentAlphaX;
            else // column
                alpha = segmentAlphaY;
            float angle = std::atan2(distance.first * std::sin(alpha), (distance.second - distance.first * std::cos(alpha)));

            if (angle > segmentTheta) {
                queue.element[end].row = neibor_row;
                queue.element[end].col = neibor_col;

                queue.cluster_size = queue.end;
                ++queue.end;

                cross_scan_flag[neibor_row] = true;
            }
        }
    }

    // check if this segment is valid
    if (queue.cluster_size >= 30) {
        ++segment_id;
    }
    else if (queue.cluster_size >= segmentValidPointNum) {
        if (std::count(cross_scan_flag, cross_scan_flag + N_SCAN, true) >= segmentValidLineNum) {
            ++segment_id;
        }
    }
    else { // segment is invalid, mark these points as outlier
        for (int i = queue.cluster_size; i >= 0; --i) {
            int idx = index_in_project_cloud(queue.element[i].row, queue.element[i].col);
            point_label_[idx] = ImageProjection::PointLabel::outlier;
            segmentation_cluster_id_.erase(idx);
        }
    }
    std::fill(cross_scan_flag, cross_scan_flag + N_SCAN, false);
    // segmentation_cluster_id_ as unordered_map, may get low performance
}

void ImageProjection::publish_cloud() {
    segmented_cloud_msg_.header = cloud_header_;
    pub_segmented_cloud_info_.publish(segmented_cloud_msg_);

    sensor_msgs::PointCloud2 laser_cloud_temp;

    // projected outlier cloud with ground
    pcl::toROSMsg(*projected_outlier_cloud_, laser_cloud_temp);
    laser_cloud_temp.header.stamp = cloud_header_.stamp;
    laser_cloud_temp.header.frame_id = "base_link";
    pub_outlier_cloud_.publish(laser_cloud_temp);

    // segmented cloud with sparse ground
    pcl::toROSMsg(*projected_ground_segment_cloud_, laser_cloud_temp);
    laser_cloud_temp.header.stamp = cloud_header_.stamp;
    laser_cloud_temp.header.frame_id = "base_link";
    pub_ground_segment_cloud_.publish(laser_cloud_temp);

    // projected full cloud
    if (pub_projected_cloud_.getNumSubscribers() > 0) {
        pcl::toROSMsg(*projected_laser_cloud_, laser_cloud_temp);
        laser_cloud_temp.header.stamp = cloud_header_.stamp;
        laser_cloud_temp.header.frame_id = "base_link";
        pub_projected_cloud_.publish(laser_cloud_temp);
    }

    // pure dense ground cloud
    if (pub_pure_ground_cloud_.getNumSubscribers() > 0) {
        pcl::toROSMsg(*projected_pure_ground_cloud_, laser_cloud_temp);
        laser_cloud_temp.header.stamp = cloud_header_.stamp;
        laser_cloud_temp.header.frame_id = "base_link";
        pub_pure_ground_cloud_.publish(laser_cloud_temp);
    }

    // segmented cloud without ground
    if (pub_pure_segmented_cloud_.getNumSubscribers() > 0) {
        pcl::toROSMsg(*projected_pure_segmented_cloud_, laser_cloud_temp);
        laser_cloud_temp.header.stamp = cloud_header_.stamp;
        laser_cloud_temp.header.frame_id = "base_link";
        pub_pure_segmented_cloud_.publish(laser_cloud_temp);
    }

    // projected full cloud with range as intensity
    if (pub_projected_cloud_with_range_.getNumSubscribers() > 0) {
        pcl::toROSMsg(*projected_cloud_with_range_, laser_cloud_temp);
        laser_cloud_temp.header.stamp = cloud_header_.stamp;
        laser_cloud_temp.header.frame_id = "base_link";
        pub_projected_cloud_with_range_.publish(laser_cloud_temp);
    }
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection ip;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();

    return 0;
}
