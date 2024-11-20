cmake_minimum_required(VERSION 3.5)
project(particle_filter)

# C++ 표준 변경
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROS2 및 종속성 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(gflags REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)


# Include 디렉토리 설정
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/lodepng
  ${rclcpp_INCLUDE_DIRS}
)

# 소스 파일 설정
set(SRCS
    src/particle_filter_node.cpp
    include/RangeLib.h
    include/RangeUtils.h
    include/CudaRangeLib.h
    include/lru_cache.h
)

# lodepng 라이브러리 추가
add_library(lodepng STATIC vendor/lodepng/lodepng.cpp)
target_include_directories(lodepng PUBLIC vendor/lodepng)

# range_lib 라이브러리 추가
add_library(range_lib ${SRCS})
ament_target_dependencies(range_lib rclcpp sensor_msgs std_msgs nav_msgs tf2_geometry_msgs lifecycle_msgs geometry_msgs tf2 tf2_ros gflags)
target_link_libraries(range_lib lodepng)

# Particle Filter 노드 추가
add_executable(particle_filter_node src/particle_filter_node.cpp)
ament_target_dependencies(particle_filter_node rclcpp sensor_msgs std_msgs nav_msgs geometry_msgs lifecycle_msgs tf2 tf2_ros gflags)
target_link_libraries(particle_filter_node range_lib ${CMAKE_THREAD_LIBS_INIT})

# 설치 경로 설정
install(TARGETS particle_filter_node range_lib lodepng
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY launch config maps
  DESTINATION share/${PROJECT_NAME}
)

# 패키지 내보내기 설정
ament_export_dependencies(rclcpp sensor_msgs std_msgs nav_msgs geometry_msgs tf2 tf2_ros gflags)
ament_package()
