#ifndef LEGO_UTILITY_H_
#define LEGO_UTILITY_H_

#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/fmt/fmt.h>

const std::string pointCloudTopic = "/velodyne_points";
const std::string imuTopic = "/imu_raw";

// Save pcd
const std::string fileDirectory = "/tmp/";

const bool loopClosureEnableFlag = false;

const int systemDelay = 0;
const int imuQueLength = 200;

#endif
