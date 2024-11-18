#include "particle_filter.h"
#include "utils.h"

ParticleFilter::ParticleFilter(ros::NodeHandle& nh) : nh_(nh) {
    // Load parameters and initialize variables
    angle_step_ = nh_.param("~angle_step", 1);
    max_particles_ = nh_.param("~max_particles", 1000);
    max_viz_particles_ = nh_.param("~max_viz_particles", 500);
    inv_squash_factor_ = 1.0 / nh_.param("~squash_factor", 1.0);
    max_range_meters_ = nh_.param("~max_range", 10.0);
    theta_discretization_ = nh_.param("~theta_discretization", 360);
    which_rm_ = nh_.param("~range_method", std::string("cddt"));
    rangelib_var_ = nh_.param("~rangelib_variant", 3);
    show_fine_timing_ = nh_.param("~fine_timing", false);
    publish_odom_ = nh_.param("~publish_odom", true);
    do_viz_ = nh_.param("~viz", true);

    // Sensor model constants
    z_short_ = nh_.param("~z_short", 0.01f);
    z_max_ = nh_.param("~z_max", 0.07f);
    z_rand_ = nh_.param("~z_rand", 0.12f);
    z_hit_ = nh_.param("~z_hit", 0.75f);
    sigma_hit_ = nh_.param("~sigma_hit", 8.0f);

    // Motion model constants
    motion_dispersion_x_ = nh_.param("~motion_dispersion_x", 0.05f);
    motion_dispersion_y_ = nh_.param("~motion_dispersion_y", 0.025f);
    motion_dispersion_theta_ = nh_.param("~motion_dispersion_theta", 0.25f);

    max_range_px_ = -1;
    odometry_data_ = {0.0, 0.0, 0.0};
    iters_ = 0;
    map_initialized_ = false;
    lidar_initialized_ = false;
    odom_initialized_ = false;
    last_pose_ = {0.0, 0.0, 0.0};
    first_sensor_update_ = true;
    local_deltas_ = std::vector<std::vector<double>>(max_particles_, std::vector<double>(3, 0.0));

    // Initialize the state
    getOccupancyMap();
    precomputeSensorModel();
    initializeGlobal();

    // Set up publishers and subscribers
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/pf/viz/inferred_pose", 1);
    particle_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/pf/viz/particles", 1);
    fake_scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>("/pf/viz/fake_scan", 1);
    rect_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/pf/viz/poly1", 1);
    if (publish_odom_) {
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/pf/pose/odom", 1);
    }

    laser_sub_ = nh_.subscribe("/scan", 1, &ParticleFilter::lidarCallback, this);
    odom_sub_ = nh_.subscribe("/odom", 1, &ParticleFilter::odomCallback, this);
    pose_sub_ = nh_.subscribe("/initialpose", 1, &ParticleFilter::clickedPoseCallback, this);
    click_sub_ = nh_.subscribe("/clicked_point", 1, &ParticleFilter::clickedPointCallback, this);

    ROS_INFO("Finished initializing, waiting on messages...");
}

void ParticleFilter::getOccupancyMap() {
    std::string map_service_name = nh_.param("~static_map", std::string("static_map"));
    ROS_INFO("Getting map from service: %s", map_service_name.c_str());
    ros::service::waitForService(map_service_name);
    nav_msgs::GetMap::Response map_msg;
    ros::ServiceClient map_client = nh_.serviceClient<nav_msgs::GetMap>(map_service_name);
    map_client.call(map_msg);

    map_info_ = map_msg.map.info;
    range_libc::PyOMap oMap(map_msg.map);
    max_range_px_ = static_cast<int>(max_range_meters_ / map_info_.resolution);

    ROS_INFO("Initializing range method: %s", which_rm_.c_str());
    if (which_rm_ == "bl") {
        range_method_ = range_libc::PyBresenhamsLine(oMap, max_range_px_);
    } else if (which_rm_ == "cddt" || which_rm_ == "pcddt") {
        range_method_ = range_libc::PyCDDTCast(oMap, max_range_px_, theta_discretization_);
        if (which_rm_ == "pcddt") {
            ROS_INFO("Pruning...");
            range_method_.prune();
        }
    } else if (which_rm_ == "rm") {
        range_method_ = range_libc::PyRayMarching(oMap, max_range_px_);
    } else if (which_rm_ == "rmgpu") {
        range_method_ = range_libc::PyRayMarchingGPU(oMap, max_range_px_);
    } else if (which_rm_ == "glt") {
        range_method_ = range_libc::PyGiantLUTCast(oMap, max_range_px_, theta_discretization_);
    }
    ROS_INFO("Done loading map");

    // Process map data
    std::vector<int8_t> map_data = map_msg.map.data;
    int map_height = map_msg.map.info.height;
    int map_width = map_msg.map.info.width;
    permissible_region_ = std::vector<std::vector<bool>>(map_height, std::vector<bool>(map_width, false));
    for (int y = 0; y < map_height; ++y) {
        for (int x = 0; x < map_width; ++x) {
            if (map_data[y * map_width + x] == 0) {
                permissible_region_[y][x] = true;
            }
        }
    }
    map_initialized_ = true;
}

void ParticleFilter::lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
    if (laser_angles_.empty()) {
        ROS_INFO("...Received first LiDAR message");
        int num_ranges = msg->ranges.size();
        laser_angles_.resize(num_ranges);
        for (int i = 0; i < num_ranges; ++i) {
            laser_angles_[i] = msg->angle_min + i * msg->angle_increment;
        }
        downsampled_angles_ = laser_angles_;
        std::vector<float> temp_angles(downsampled_angles_.begin(), downsampled_angles_.end());
        downsampled_angles_ = temp_angles;
    }
    downsampled_ranges_ = std::vector<float>(msg->ranges.begin(), msg->ranges.end());
    lidar_initialized_ = true;
}

void ParticleFilter::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    std::vector<double> position = {msg->pose.pose.position.x, msg->pose.pose.position.y};
    double orientation = Utils::quaternionToAngle(msg->pose.pose.orientation);
    std::vector<double> pose = {position[0], position[1], orientation};

    if (!last_pose_.empty()) {
        std::vector<std::vector<double>> rotation_matrix = Utils::rotationMatrix(-last_pose_[2]);
        std::vector<double> delta = {position[0] - last_pose_[0], position[1] - last_pose_[1]};
        std::vector<double> local_delta = {rotation_matrix[0][0] * delta[0] + rotation_matrix[0][1] * delta[1],
                                           rotation_matrix[1][0] * delta[0] + rotation_matrix[1][1] * delta[1]};
        odometry_data_ = {local_delta[0], local_delta[1], orientation - last_pose_[2]};
        last_pose_ = pose;
        last_stamp_ = msg->header.stamp;
        odom_initialized_ = true;
    } else {
        ROS_INFO("...Received first Odometry message");
        last_pose_ = pose;
    }

    update();
}

void ParticleFilter::clickedPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
    initializeParticlesPose(msg->pose.pose);
}

void ParticleFilter::clickedPointCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
    initializeGlobal();
}

void ParticleFilter::initializeParticlesPose(const geometry_msgs::Pose& pose) {
    ROS_INFO("SETTING POSE");
    state_mutex_.lock();
    weights_ = std::vector<double>(max_particles_, 1.0 / max_particles_);
    particles_ = std::vector<std::vector<double>>(max_particles_, std::vector<double>(3, 0.0));
    for (int i = 0; i < max_particles_; ++i) {
        particles_[i][0] = pose.position.x + Utils::randomNormal(0.0, 0.5);
        particles_[i][1] = pose.position.y + Utils::randomNormal(0.0, 0.5);
        particles_[i][2] = Utils::quaternionToAngle(pose.orientation) + Utils::randomNormal(0.0, 0.4);
    }
    state_mutex_.unlock();
}

void ParticleFilter::initializeGlobal() {
    ROS_INFO("GLOBAL INITIALIZATION");
    state_mutex_.lock();
    std::vector<int> permissible_x, permissible_y;
    for (int y = 0; y < permissible_region_.size(); ++y) {
        for (int x = 0; x < permissible_region_[y].size(); ++x) {
            if (permissible_region_[y][x]) {
                permissible_x.push_back(x);
                permissible_y.push_back(y);
            }
        }
    }

    particles_ = std::vector<std::vector<double>>(max_particles_, std::vector<double>(3, 0.0));
    for (int i = 0; i < max_particles_; ++i) {
        int index = rand() % permissible_x.size();
        particles_[i][0] = permissible_x[index];
        particles_[i][1] = permissible_y[index];
        particles_[i][2] = static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;
    }

    Utils::mapToWorld(particles_, map_info_);
    weights_ = std::vector<double>(max_particles_, 1.0 / max_particles_);
    state_mutex_.unlock();
}

// Define the remaining methods (precomputeSensorModel, motionModel, sensorModel, monteCarloLocalization, expectedPose, update, publishTransform, visualize, publishParticles, publishScan) similarly

int main(int argc, char** argv) {
    ros::init(argc, argv, "particle_filter");
    ros::NodeHandle nh;

    ParticleFilter pf(nh);
    ros::spin();

    return 0;
}
