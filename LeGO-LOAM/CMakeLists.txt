cmake_minimum_required(VERSION 2.8.3)
project(lego_loam)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -std=c++14")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  pcl_ros
  pcl_conversions

  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cloud_msgs
)

find_package(glog 0.5.0 REQUIRED QUIET)
find_package(Eigen3 REQUIRED QUIET)
find_package(spdlog REQUIRED QUIET)
find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(Ceres REQUIRED QUIET)
find_package(Boost REQUIRED COMPONENTS
  serialization
  thread
  timer
  chrono
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS cloud_msgs
  DEPENDS PCL
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${Eigen_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
	${spdlog_INCLUDE_DIR}
	${CERES_INCLUDE_DIR}
)

link_directories(
	include
	${PCL_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}
	${Boost_LIBRARY_DIRS}
	${spdlog_LIBRARY_DIRS}
	${CERES_LIBRARY_DIRS}
)

set(helper_srcs src/ceres_wrapper.cpp
    src/eigen_wrapper.cpp
    src/lego_math.cpp)

add_library(lego_helper ${helper_srcs})
target_link_libraries(lego_helper ${catkin_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen glog::glog ${CERES_LIBRARIES})

add_executable(imageProjection src/imageProjection.cpp)
add_dependencies(imageProjection ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(imageProjection lego_helper ${catkin_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen spdlog::spdlog)

add_executable(featureAssociation src/featureAssociation.cpp)
add_dependencies(featureAssociation ${catkin_EXPORTED_TARGETS} cloud_msgs_gencpp)
target_link_libraries(featureAssociation lego_helper ${catkin_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen spdlog::spdlog)

add_executable(mapOptmization src/mapOptmization.cpp)
target_link_libraries(mapOptmization lego_helper ${catkin_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen ${Boost_LIBRARIES} gtsam spdlog::spdlog)

# add_executable(transformFusion src/transformFusion.cpp)
# target_link_libraries(transformFusion lego_helper ${catkin_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen spdlog::spdlog)

