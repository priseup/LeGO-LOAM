#ifndef LEGO_LASER_INFO_H_
#define LEGO_LASER_INFO_H_

// Using velodyne cloud "ring" channel for image projection (other lidar may have different name for this channel, change "PointXYZIR" below)
const bool useCloudRing = true; // if true, laser_resolution_vertical and ang_bottom are not used

// VLP-16
const int N_SCAN = 16;
const int Horizon_SCAN = 1800;
const float laser_resolution_horizon = 0.2;
const float laser_resolution_vertical = 2.0;
const float ang_bottom = 15.0+0.1;
const int groundScanInd = 7;

// HDL-32E
// extern const int N_SCAN = 32;
// extern const int Horizon_SCAN = 1800;
// extern const float laser_resolution_horizon = 360.0/float(Horizon_SCAN);
// extern const float laser_resolution_vertical = 41.33/float(N_SCAN-1);
// extern const float ang_bottom = 30.67;
// extern const int groundScanInd = 20;

// VLS-128
// extern const int N_SCAN = 128;
// extern const int Horizon_SCAN = 1800;
// extern const float laser_resolution_horizon = 0.2;
// extern const float laser_resolution_vertical = 0.3;
// extern const float ang_bottom = 25.0;
// extern const int groundScanInd = 10;

// Ouster users may need to uncomment line 159 in imageProjection.cpp
// Usage of Ouster imu data is not fully supported yet (LeGO-LOAM needs 9-DOF IMU), please just publish point cloud data
// Ouster OS1-16
// extern const int N_SCAN = 16;
// extern const int Horizon_SCAN = 1024;
// extern const float laser_resolution_horizon = 360.0/float(Horizon_SCAN);
// extern const float laser_resolution_vertical = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 7;

// Ouster OS1-64
// extern const int N_SCAN = 64;
// extern const int Horizon_SCAN = 1024;
// extern const float laser_resolution_horizon = 360.0/float(Horizon_SCAN);
// extern const float laser_resolution_vertical = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 15;

const float scanPeriod = 0.1;

const int points_num = N_SCAN * Horizon_SCAN;

#endif  // LEGO_LASER_INFO_H_
