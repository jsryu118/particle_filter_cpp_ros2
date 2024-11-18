#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <mutex>
#include <random> 
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/srv/get_map.hpp>
#include "utils.h"
#include "RangeLib.h"
#include "lifecycle_msgs/srv/get_state.hpp"
#include "lifecycle_msgs/srv/change_state.hpp"

class ParticleFilter : public rclcpp::Node {
public:
    ParticleFilter();
    void lidarCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void clickedPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void clickedPointCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg);

private:
    void declare_and_get_parameters();
    void getOccupancyMap();
    void publish_tf(const std::vector<double>& x_y_yaw, const rclcpp::Time& stamp);
    void visualize();
    void publishParticles(const std::vector<std::vector<double>>& particles);
    void publishScan(const std::vector<float>& angles, const std::vector<float>& ranges);
    void initializeParticlesPose(const geometry_msgs::msg::Pose& pose);
    void initializeGlobal();
    void precomputeSensorModel();
    void motionModel(std::vector<std::vector<double>>& proposal_dist, const std::vector<double>& action);
    void sensorModel(std::vector<std::vector<double>>& proposal_dist, const std::vector<float>& obs, std::vector<double>& weights);
    void monteCarloLocalization(const std::vector<double>& action, const std::vector<float>& obs);
    std::vector<double> expectedPose();
    void update();
    int randomIndex(const std::vector<double>& weights);

    // Node and ROS2 components
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr fake_scan_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr clicked_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr clicked_point_sub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::mutex state_mutex_;
    std::mt19937 gen_;

    std::default_random_engine generator_;
    Utils::CircularArray smoothing_;
    Utils::Timer timer_;
    rclcpp::Time t_start_, t_propose_, t_motion_, t_sensor_, t_norm_;
    double t_total_;

    // Topic parameters
    std::string map_topic_;
    std::string sub_scan_topic_;
    std::string sub_wheel_odom_topic_;
    std::string sub_imu_odom_topic_;

    // Range method parameters
    std::unique_ptr<ranges::RangeMethod> range_method_;

    // ROS parameters
    int odom_idx_;
    bool odom_fast_;
    bool no_initial_guess_;
    int angle_step_;
    int num_downsampled_ranges_;
    int max_particles_;
    int max_viz_particles_;
    float inv_squash_factor_;
    float max_range_meters_;
    int theta_discretization_;
    std::string which_rm_;
    int rangelib_var_;
    bool show_fine_timing_;
    bool publish_odom_;
    bool do_viz_;

    float z_short_;
    float z_max_;
    float z_rand_;
    float z_hit_;
    float sigma_hit_;
    float motion_dispersion_x_;
    float motion_dispersion_y_;
    float motion_dispersion_theta_;

    float init_pose_sigma_x_;
    float init_pose_sigma_y_;
    float init_pose_sigma_yaw_;
    float init_point_sigma_x_;
    float init_point_sigma_y_;
    float init_point_sigma_yaw_;

    int VAR_NO_EVAL_SENSOR_MODEL = 0;
    int VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1;
    int VAR_REPEAT_ANGLES_EVAL_SENSOR = 2;
    int VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3;
    int VAR_RADIAL_CDDT_OPTIMIZATIONS = 4;

    int max_range_px_;
    std::vector<double> odometry_data_;
    std::vector<float> laser_;
    int iters_;
    nav_msgs::msg::MapMetaData map_info_;
    bool map_initialized_;
    bool lidar_initialized_;
    bool odom_initialized_;
    std::vector<double> last_pose_;
    std::vector<double> laser_angles_;
    std::vector<float> downsampled_angles_;
    std::vector<double> downsampled_ranges_;
    ranges::OMap* omap_ptr_;
    std::vector<std::vector<bool>> permissible_region_;

    rclcpp::Time last_time_;
    rclcpp::Time last_stamp_;
    bool first_sensor_update_;
    std::vector<std::vector<double>> local_deltas_;
    std::vector<std::vector<float>> queries_;
    std::vector<float> fake_ranges_;
    std::vector<float> tiled_angles_;
    std::vector<std::vector<double>> sensor_model_table_;

    std::vector<double> inferred_pose_;
    std::vector<int> particle_indices_;
    std::vector<std::vector<double>> particles_;
    std::vector<double> weights_;

    std::string sub_scan_topic_frame_;

};

#endif // PARTICLE_FILTER_H
