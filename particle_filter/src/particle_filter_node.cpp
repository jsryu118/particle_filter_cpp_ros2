#include "particle_filter_node.h"
#include "utils.h"


ParticleFilter::ParticleFilter() : Node("particle_filter"), smoothing_(10), timer_(10), state_mutex_(), gen_(std::random_device{}()) {
    declare_and_get_parameters();
    // Initialize publishers and subscribers
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 1);
    particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 1);
    fake_scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/pf/viz/fake_scan", 1);

    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        sub_scan_topic_, 1, std::bind(&ParticleFilter::lidarCallback, this, std::placeholders::_1));

    // Subscribe to RViz2's "2D Pose Estimate" and "2D Nav Goal" topics
    clicked_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 10, std::bind(&ParticleFilter::clickedPoseCallback, this, std::placeholders::_1));

    clicked_point_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/clicked_point", 10, std::bind(&ParticleFilter::clickedPointCallback, this, std::placeholders::_1));

    if (odom_idx_ == 0) {
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            sub_wheel_odom_topic_, 1, std::bind(&ParticleFilter::odomCallback, this, std::placeholders::_1));
    } else if (odom_idx_ == 1) {
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            sub_imu_odom_topic_, 1, std::bind(&ParticleFilter::odomCallback, this, std::placeholders::_1));
    }

    // Initialize the state
    getOccupancyMap();
    particles_ = std::vector<std::vector<double>>(max_particles_, std::vector<double>(3, 0.0));
    precomputeSensorModel();
    initializeGlobal();
}

void ParticleFilter::declare_and_get_parameters() {
    // Declare and load parameters in one step
    map_topic_ = declare_parameter<std::string>("map_topic", "/pf_map");
    sub_scan_topic_ = declare_parameter<std::string>("scan_topic", "/scan");
    sub_wheel_odom_topic_ = declare_parameter<std::string>("vesc_odometry_topic", "/vesc/odom");
    sub_imu_odom_topic_ = declare_parameter<std::string>("imu_odometry_topic", "/imu/odom");

    odom_fast_ = declare_parameter<bool>("odom_faster_than_lidar", true);
    no_initial_guess_ = declare_parameter<bool>("no_initial_guess", false);
    angle_step_ = declare_parameter<int>("angle_step", 18);
    max_particles_ = declare_parameter<int>("max_particles", 4000);
    max_viz_particles_ = declare_parameter<int>("max_viz_particles", 60);
    inv_squash_factor_ = 1.0 / declare_parameter<double>("squash_factor", 2.2);
    max_range_meters_ = declare_parameter<double>("max_range", 10.0);
    theta_discretization_ = declare_parameter<int>("theta_discretization", 112);
    which_rm_ = declare_parameter<std::string>("range_method", "cddt");
    rangelib_var_ = declare_parameter<int>("rangelib_variant", 2);
    show_fine_timing_ = declare_parameter<bool>("fine_timing", false);
    publish_odom_ = declare_parameter<bool>("publish_odom", true);
    do_viz_ = declare_parameter<bool>("viz", true);

    z_short_ = declare_parameter<double>("z_short", 0.01);
    z_max_ = declare_parameter<double>("z_max", 0.07);
    z_rand_ = declare_parameter<double>("z_rand", 0.12);
    z_hit_ = declare_parameter<double>("z_hit", 0.75);
    sigma_hit_ = declare_parameter<double>("sigma_hit", 8.0);

    motion_dispersion_x_ = declare_parameter<double>("motion_dispersion_x", 0.05);
    motion_dispersion_y_ = declare_parameter<double>("motion_dispersion_y", 0.025);
    motion_dispersion_theta_ = declare_parameter<double>("motion_dispersion_theta", 0.25);

    init_pose_sigma_x_ = declare_parameter<double>("initialize_pose_sigma_x", 0.5);
    init_pose_sigma_y_ = declare_parameter<double>("initialize_pose_sigma_y", 0.5);
    init_pose_sigma_yaw_ = declare_parameter<double>("initialize_pose_sigma_yaw", 0.4);

    odom_idx_ = declare_parameter<int>("odom_idx", 0);
}

void ParticleFilter::getOccupancyMap() {
    std::string map_service_name = "/map_server/map";
    RCLCPP_INFO(this->get_logger(), "Getting map from service: %s", map_service_name.c_str());

    // Create service clients
    auto map_client = this->create_client<nav_msgs::srv::GetMap>(map_service_name);
    auto get_state_client = this->create_client<lifecycle_msgs::srv::GetState>("/map_server/get_state");
    auto change_state_client = this->create_client<lifecycle_msgs::srv::ChangeState>("/map_server/change_state");

    // Wait for lifecycle services to be available
    while (!get_state_client->wait_for_service(std::chrono::seconds(1)) ||
           !change_state_client->wait_for_service(std::chrono::seconds(1))) {
        RCLCPP_WARN(this->get_logger(), "Waiting for lifecycle services to become available...");
    }

    auto state_request = std::make_shared<lifecycle_msgs::srv::GetState::Request>();

    // Define lifecycle states for ROS 2 Foxy
    constexpr uint8_t UNCONFIGURED = 1;
    constexpr uint8_t INACTIVE = 2;
    constexpr uint8_t ACTIVE = 3;

    // Loop to check and change state if necessary
    while (rclcpp::ok()) {
        // Get current state of the map server
        auto state_future = get_state_client->async_send_request(state_request);

        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), state_future) ==
            rclcpp::FutureReturnCode::SUCCESS) {
            auto state_response = state_future.get();

            // If the map server is already active, exit the loop
            if (state_response->current_state.id == ACTIVE) {
                RCLCPP_INFO(this->get_logger(), "Map server is now active.");
                break;
            }

            // If the map server is unconfigured, send a configure request
            if (state_response->current_state.id == UNCONFIGURED) {
                RCLCPP_WARN(this->get_logger(), "Map server is unconfigured. Sending configure request...");
                auto configure_request = std::make_shared<lifecycle_msgs::srv::ChangeState::Request>();
                configure_request->transition.id = lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE;
                auto configure_future = change_state_client->async_send_request(configure_request);
                rclcpp::spin_until_future_complete(this->get_node_base_interface(), configure_future);
            }

            // If the map server is inactive, send an activate request
            if (state_response->current_state.id == INACTIVE) {
                RCLCPP_WARN(this->get_logger(), "Map server is inactive. Sending activate request...");
                auto activate_request = std::make_shared<lifecycle_msgs::srv::ChangeState::Request>();
                activate_request->transition.id = lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE;
                auto activate_future = change_state_client->async_send_request(activate_request);
                rclcpp::spin_until_future_complete(this->get_node_base_interface(), activate_future);
            }

        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to get state of map server.");
        }

        // Sleep for a short duration to prevent rapid looping
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Request the map after ensuring the map server is active
    auto request = std::make_shared<nav_msgs::srv::GetMap::Request>();
    auto future = map_client->async_send_request(request);

    // Wait for the response and process it
    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
        rclcpp::FutureReturnCode::SUCCESS) {
        auto response = future.get();
        RCLCPP_INFO(this->get_logger(), "Successfully received map data.");
        map_info_ = response->map.info;
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to get map data.");
    }

    // Process the response when the service call is complete
    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
        rclcpp::FutureReturnCode::SUCCESS) {
        auto response = future.get();

        // Map information
        map_info_ = response->map.info;
        std_msgs::msg::Header map_header = response->map.header;
        map_topic_ = map_header.frame_id;

        int width = map_info_.width;
        int height = map_info_.height;

        // Initialize the occupancy map
        omap_ptr_ = new ranges::OMap(height, width);

        // Process the map data (0: free, >10: blocked)
        std::vector<int8_t> map_data = response->map.data;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                if (map_data[h * width + w] > 10) {
                    omap_ptr_->grid[h][w] = true;
                }
            }
        }

        // Cache constants for coordinate space conversion
        double angle = -1.0 * Utils::quaternionToAngle(map_info_.origin.orientation);
        omap_ptr_->world_scale = map_info_.resolution;
        omap_ptr_->world_angle = angle;
        omap_ptr_->world_origin_x = map_info_.origin.position.x;
        omap_ptr_->world_origin_y = map_info_.origin.position.y;
        omap_ptr_->world_sin_angle = sin(angle);
        omap_ptr_->world_cos_angle = cos(angle);

        max_range_px_ = static_cast<int>(max_range_meters_ / map_info_.resolution);
        RCLCPP_INFO(this->get_logger(), "Initializing range method: %s", which_rm_.c_str());

        // Initialize range method
        if (which_rm_ == "bl") {
            range_method_ = std::unique_ptr<ranges::RangeMethod>(new ranges::BresenhamsLine(*omap_ptr_, max_range_px_));
       } else if (which_rm_ == "cddt" || which_rm_ == "pcddt") {
            auto cddt_cast = new ranges::CDDTCast(*omap_ptr_, max_range_px_, theta_discretization_);
            if (which_rm_ == "pcddt") {
                RCLCPP_INFO(this->get_logger(), "Pruning...");
                cddt_cast->prune(max_range_px_);
            }
            range_method_ = std::unique_ptr<ranges::RangeMethod>(cddt_cast);
        } else if (which_rm_ == "rm") {
            range_method_ = std::unique_ptr<ranges::RangeMethod>(new ranges::RayMarching(*omap_ptr_, max_range_px_));
        } else if (which_rm_ == "rmgpu") {
            range_method_ = std::unique_ptr<ranges::RangeMethod>(new ranges::RayMarchingGPU(*omap_ptr_, max_range_px_));
        } else if (which_rm_ == "glt") {
            range_method_ = std::unique_ptr<ranges::RangeMethod>(new ranges::GiantLUTCast(*omap_ptr_, max_range_px_, theta_discretization_));
        }
        RCLCPP_INFO(this->get_logger(), "Done loading map");

        // 0: not permissible, 1: permissible
        permissible_region_ = std::vector<std::vector<bool>>(height, std::vector<bool>(width, false));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (map_data[y * width + x] == 0) {
                    permissible_region_[y][x] = true;
                }
            }
        }
        map_initialized_ = true;

    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to call service: %s", map_service_name.c_str());
    }

}

void ParticleFilter::publish_tf(const std::vector<double> &x_y_yaw, const rclcpp::Time &stamp) {
    // Ensure map and LiDAR are initialized
    if (map_initialized_ && lidar_initialized_) {
        double x = x_y_yaw[0];
        double y = x_y_yaw[1];
        double yaw = x_y_yaw[2];

        // Create transform
        geometry_msgs::msg::TransformStamped transform_stamped;
        transform_stamped.header.stamp = this->now();
        transform_stamped.header.frame_id = map_topic_;
        transform_stamped.child_frame_id = sub_scan_topic_frame_;
        transform_stamped.transform.translation.x = x;
        transform_stamped.transform.translation.y = y;
        transform_stamped.transform.translation.z = 0.0;

        // Create quaternion from yaw
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        transform_stamped.transform.rotation.x = q.x();
        transform_stamped.transform.rotation.y = q.y();
        transform_stamped.transform.rotation.z = q.z();
        transform_stamped.transform.rotation.w = q.w();

        // Broadcast the transform
        tf_broadcaster_->sendTransform(transform_stamped);

        // Publish Odometry message if enabled
        if (publish_odom_) {
            nav_msgs::msg::Odometry odom;
            odom.header.stamp = this->now();
            odom.header.frame_id = map_topic_;
            odom.child_frame_id = sub_scan_topic_frame_;
            odom.pose.pose.position.x = x;
            odom.pose.pose.position.y = y;
            odom.pose.pose.position.z = 0.0;
            odom.pose.pose.orientation.x = q.x();
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z();
            odom.pose.pose.orientation.w = q.w();

            odom_pub_->publish(odom);
        }
    } else {
        RCLCPP_INFO(this->get_logger(), "Map or LiDAR have not been initialized!");
    }
}


void ParticleFilter::visualize() {
    if (!do_viz_) {
        return;
    }

//     if (pose_pub_.getNumSubscribers() > 0 && !inferred_pose_.empty()) {
//         // Publish the inferred pose for visualization
//         geometry_msgs::PoseStamped ps;
//         ps.header = Utils::makeHeader(map_topic_frame_);
//         ps.pose.position.x = inferred_pose_[0];
//         ps.pose.position.y = inferred_pose_[1];
//         ps.pose.orientation = Utils::angleToQuaternion(inferred_pose_[2]);
//         pose_pub_.publish(ps);
//     }

//     if (particle_pub_.getNumSubscribers() > 0) {
//         // std::cout<<"particle"<<std::endl;
//         // Publish a downsampled version of the particle distribution to avoid a lot of latency
//         if (max_particles_ > max_viz_particles_) {
//             // Randomly downsample particles
//             // particle_indices_.resize(max_particles_);
            
//             // std::cout<<particle_indices_<<std::endl;
//             std::vector<int> proposal_indices = Utils::randomChoice(max_particles_, max_viz_particles_, weights_);
//             std::vector<std::vector<double>> sampled_particles(max_viz_particles_, std::vector<double>(3, 0.0));
//             for (size_t i = 0; i < proposal_indices.size(); ++i) {
//                 sampled_particles[i] = particles_[proposal_indices[i]];
//             }
//             publishParticles(sampled_particles);
//         } else {
//             publishParticles(particles_);
//         }
//     }

//     if (fake_scan_pub_.getNumSubscribers() > 0 && !fake_ranges_.empty()) {
//         std::cout<<"fake_scan"<<std::endl;
//         // Generate the scan from the point of view of the inferred position for visualization
//         for (size_t i = 0; i < queries_.size(); ++i) {
//             queries_[i][0] = inferred_pose_[0];
//             queries_[i][1] = inferred_pose_[1];
//             queries_[i][2] = downsampled_angles_[i] + inferred_pose_[2];
//         }

//         std::vector<float> flat_queries;
//         for (const auto& query : queries_) {
//             flat_queries.insert(flat_queries.end(), query.begin(), query.end());
//         }

//         range_method_->numpy_calc_range(flat_queries.data(), fake_ranges_.data(), fake_ranges_.size());
//         // std::cout<<fake_ranges_.data()<<std::endl;
//         publishScan(downsampled_angles_, fake_ranges_);
//     }
}

// void ParticleFilter::publishParticles(const std::vector<std::vector<double>>& particles) {
//     // Publish the given particles as a PoseArray object
//     geometry_msgs::PoseArray pa;
//     pa.header = Utils::makeHeader("map");
//     pa.poses = Utils::particlesToPoses(particles);
//     particle_pub_->publish(pa);
// }

// void ParticleFilter::publishScan(const std::vector<float>& angles, const std::vector<float>& ranges) {
//     // Publish the given angles and ranges as a laser scan message
//     sensor_msgs::LaserScan ls;
//     ls.header = Utils::makeHeader("laser", last_stamp_);
//     ls.angle_min = *std::min_element(angles.begin(), angles.end());
//     ls.angle_max = *std::max_element(angles.begin(), angles.end());
//     ls.angle_increment = std::abs(angles[0] - angles[1]);
//     ls.range_min = 0;
//     ls.range_max = *std::max_element(ranges.begin(), ranges.end());
//     ls.ranges = ranges;
//     fake_scan_pub_->publish(ls);
// }

void ParticleFilter::lidarCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg){
    if (laser_angles_.empty()) {
        std_msgs::msg::Header laser_header = msg->header;
        sub_scan_topic_frame_ = laser_header.frame_id;
        rclcpp::Time map_timestamp = laser_header.stamp;
        RCLCPP_INFO(this->get_logger(), "...Received first LiDAR message");

        int num_ranges = msg->ranges.size();
        laser_angles_.resize(num_ranges);
        for (int i = 0; i < num_ranges; ++i) {
            laser_angles_[i] = msg->angle_min + i * msg->angle_increment;
        }

        num_downsampled_ranges_ = num_ranges / angle_step_;
        if (num_ranges % angle_step_ != 0){num_downsampled_ranges_ += 1;}
        RCLCPP_INFO(this->get_logger(), "The number of downsampled ranges is: %u", num_downsampled_ranges_);
        downsampled_angles_.resize(num_downsampled_ranges_);
        downsampled_ranges_.resize(num_downsampled_ranges_);
        queries_ = std::vector<std::vector<float>>(num_downsampled_ranges_, std::vector<float>(3, 0.0));

        for (int i = 0; i < num_downsampled_ranges_; ++i) {
            downsampled_angles_[i] = laser_angles_[i*angle_step_];
        }
        lidar_initialized_ = true;
    }

    for (int i = 0; i < num_downsampled_ranges_; ++i) {
        downsampled_ranges_[i] = msg->ranges[i*angle_step_];
    }
    
    if (odom_fast_){
        if (no_initial_guess_){
            //// TODO
        }
        update();
    }
    
}



void ParticleFilter::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::vector<double> position = {msg->pose.pose.position.x, msg->pose.pose.position.y};
    double orientation = Utils::quaternionToAngle(msg->pose.pose.orientation);
    std::vector<double> pose = {position[0], position[1], orientation};

    if (!last_pose_.empty()) {
        std::vector<std::vector<double>> rotation_matrix = Utils::rotationMatrix(-last_pose_[2]);
        std::vector<double> delta = {position[0] - last_pose_[0], position[1] - last_pose_[1]};
        std::vector<double> local_delta = {
            rotation_matrix[0][0] * delta[0] + rotation_matrix[0][1] * delta[1],
            rotation_matrix[1][0] * delta[0] + rotation_matrix[1][1] * delta[1]
        };
        odometry_data_ = {local_delta[0], local_delta[1], orientation - last_pose_[2]};
        last_pose_ = pose;
        last_stamp_ = rclcpp::Time(msg->header.stamp);  // ROS 2에서 타임스탬프 변환
    } else {
        RCLCPP_INFO(this->get_logger(), "...Received first Odometry message");
        last_pose_ = pose;
        odom_initialized_ = true;
    }

    if (!odom_fast_) {
        update();
    }
}

void ParticleFilter::clickedPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg){
    initializeParticlesPose(msg->pose.pose);
}

void ParticleFilter::clickedPointCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg){
    initializeGlobal();
}



void ParticleFilter::initializeParticlesPose(const geometry_msgs::msg::Pose& pose) {
    RCLCPP_INFO(this->get_logger(), "SETTING POSE");
    state_mutex_.lock();
    weights_ = std::vector<double>(max_particles_, 1.0 / max_particles_);
        // std::cout<<weights_[0]<<std::endl;
    
    for (int i = 0; i < max_particles_; ++i) {
        particles_[i][0] = pose.position.x + Utils::randomNormal(0.0, init_pose_sigma_x_);
        particles_[i][1] = pose.position.y + Utils::randomNormal(0.0, init_pose_sigma_y_);
        particles_[i][2] = Utils::quaternionToAngle(pose.orientation) + Utils::randomNormal(0.0, init_pose_sigma_yaw_);
    }
    state_mutex_.unlock();
}

void ParticleFilter::initializeGlobal() {
    RCLCPP_INFO(this->get_logger(), "GLOBAL INITIALIZATION");
    std::lock_guard<std::mutex> lock(state_mutex_);
    std::vector<int> permissible_x, permissible_y;
    for (int y = 0; y < permissible_region_.size(); ++y) {
        for (int x = 0; x < permissible_region_[y].size(); ++x) {
            if (permissible_region_[y][x]) {
                permissible_x.push_back(x);
                permissible_y.push_back(y);
            }
        }
    }
    std::uniform_int_distribution<int> dist(0, permissible_x.size() - 1);
    
    for (int i = 0; i < max_particles_; ++i) {
        // int index = rand() % permissible_x.size();
        int index = dist(gen_);

        particles_[i][0] = permissible_x[index];
        particles_[i][1] = permissible_y[index];
        particles_[i][2] = std::uniform_real_distribution<double>(0.0, 2.0 * M_PI)(gen_);
    }
    // Utils::mapToWorld(particles_, map_info_);
    // weights_ = std::vector<double>(max_particles_, 1.0 / max_particles_);
}

void ParticleFilter::precomputeSensorModel() {
    double z_short = z_short_;
    double z_max = z_max_;
    double z_rand = z_rand_;
    double z_hit = z_hit_;
    double sigma_hit = sigma_hit_;

    int table_width = static_cast<int>(max_range_meters_) + 1;
    sensor_model_table_ = std::vector<std::vector<double>>(table_width, std::vector<double>(table_width, 0.0));

    for (int d = 0; d < table_width; ++d) {
        double norm = 0.0;

        for (int r = 0; r < table_width; ++r) {
            double prob = 0.0;
            double z = static_cast<double>(r - d);

            // reflects from the intended object
            prob += z_hit * std::exp(-(z * z) / (2.0 * sigma_hit * sigma_hit)) / (sigma_hit * std::sqrt(2.0 * M_PI));

            // observed range is less than the predicted range - short reading
            if (r < d) {
                prob += 2.0 * z_short * (d - r) / static_cast<double>(d);
            }

            // erroneous max range measurement
            if (r == static_cast<int>(max_range_meters_)) {
                prob += z_max;
            }

            // random measurement
            if (r < static_cast<int>(max_range_meters_)) {
                prob += z_rand * 1.0 / static_cast<double>(max_range_meters_);
            }

            norm += prob;
            sensor_model_table_[r][d] = prob;
        }

        // normalize
        for (int r = 0; r < table_width; ++r) {
            sensor_model_table_[r][d] /= norm;
        }
    }

    // Flatten the sensor model table for passing to set_sensor_model
    std::vector<double> flattened_table;
    flattened_table.reserve(table_width * table_width);
    for (const auto& row : sensor_model_table_) {
        flattened_table.insert(flattened_table.end(), row.begin(), row.end());
    }

    // Upload the sensor model to RangeLib for ultra fast resolution
    if (rangelib_var_ > 0) {
        range_method_->set_sensor_model(flattened_table.data(), table_width);
    }
    RCLCPP_INFO(this->get_logger(), "Finish precompute sensor model!!!!");
}

void ParticleFilter::monteCarloLocalization(const std::vector<double>& initial_guess, const std::vector<float>& real_scan) {
    // If fine timing is enabled, start the timer
    if (show_fine_timing_) {
        t_start_ = this->now();
    }

    // Resampling: Draw the proposal distribution from the old particles
    std::vector<std::vector<double>> new_particles(max_particles_, std::vector<double>(3, 0.0));
    for (int i = 0; i < max_particles_; ++i) {
        int index = randomIndex(weights_);
        new_particles[i] = particles_[index];
    }

    // Record the time after proposal distribution
    if (show_fine_timing_) {
        t_propose_ = this->now();
    }

    // Compute the motion model to update the proposal distribution
    motionModel(new_particles, initial_guess);

    // Record the time after motion model update
    if (show_fine_timing_) {
        t_motion_ = this->now();
    }

    // Compute the sensor model
    sensorModel(new_particles, real_scan, weights_);

    // Record the time after sensor model update
    if (show_fine_timing_) {
        t_sensor_ = this->now();
    }

    // Normalize importance weights
    double sum_weights = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    for (auto& weight : weights_) {
        weight /= sum_weights;
    }

    // Record the time after normalization
    if (show_fine_timing_) {
        t_norm_ = this->now();
        t_total_ = (t_norm_ - t_start_).seconds() / 100.0;
    }

    // If fine timing is enabled and every 10 iterations, log the timing information
    if (show_fine_timing_ && iters_ % 10 == 0) {
        RCLCPP_INFO(this->get_logger(), "MCL: propose: %.2f motion: %.2f sensor: %.2f norm: %.2f",
                    (t_propose_ - t_start_).seconds() / t_total_,
                    (t_motion_ - t_propose_).seconds() / t_total_,
                    (t_sensor_ - t_motion_).seconds() / t_total_,
                    (t_norm_ - t_sensor_).seconds() / t_total_);
    }

    // Save the particles
    particles_ = new_particles;
}

int ParticleFilter::randomIndex(const std::vector<double>& weights) {
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return dist(generator_);
}


void ParticleFilter::motionModel(std::vector<std::vector<double>>& proposal_dist, const std::vector<double>& initial_guess) {
    for (int i = 0; i < max_particles_; ++i) {
        double cos_ = cos(proposal_dist[i][2]);
        double sin_ = sin(proposal_dist[i][2]);

        local_deltas_[i][0] = cos_ * initial_guess[0] - sin_ * initial_guess[1];
        local_deltas_[i][1] = sin_ * initial_guess[0] + cos_ * initial_guess[1];
        local_deltas_[i][2] = initial_guess[2];

        proposal_dist[i][0] += local_deltas_[i][0] + Utils::randomNormal(0.0, motion_dispersion_x_);
        proposal_dist[i][1] += local_deltas_[i][1] + Utils::randomNormal(0.0, motion_dispersion_y_);
        proposal_dist[i][2] += local_deltas_[i][2] + Utils::randomNormal(0.0, motion_dispersion_theta_);
    }
}


// // void ParticleFilter::sensorModel(std::vector<std::vector<double>>& proposal_dist, const std::vector<float>& obs, std::vector<double>& weights) {
//     // int num_rays = downsampled_angles_.size();
//     // std::vector<float> flat_obs;
//     //     for (const auto& ob : obs) {
//     //         flat_queries.insert(flat_queries.end(), ob.begin(), ob.end());
//     //     }
//     // // Only allocate buffers once to avoid slowness
//     // if (first_sensor_update_) {
//     //     if (rangelib_var_ <= 1) {
//     //         queries_ = std::vector<std::vector<float>>(num_rays * max_particles_, std::vector<float>(3, 0.0));
//     //     } else {
//     //         queries_ = std::vector<std::vector<float>>(max_particles_, std::vector<float>(3, 0.0));
//     //     }
//     //     ranges_ = std::vector<float>(num_rays * max_particles_, 0.0f);
//     //     tiled_angles_ = std::vector<float>(num_rays * max_particles_);
//     //     for (int i = 0; i < max_particles_; ++i) {
//     //         std::copy(downsampled_angles_.begin(), downsampled_angles_.end(), tiled_angles_.begin() + i * num_rays);
//     //     }
//     //     first_sensor_update_ = false;
//     // }

//     // if (rangelib_var_ == VAR_RADIAL_CDDT_OPTIMIZATIONS) {
//     //     if (which_rm_ == "cddt" || which_rm_ == "pcddt") {
//     //         for (int i = 0; i < max_particles_; ++i) {
//     //             queries_[i][0] = proposal_dist[i][0];
//     //             queries_[i][1] = proposal_dist[i][1];
//     //             queries_[i][2] = proposal_dist[i][2];
//     //         }
//     //         range_method_->calc_range_many_radial_optimized(num_rays, downsampled_angles_[0], downsampled_angles_.back(), queries_, ranges_);
//     //         range_method_->eval_sensor_model(flat_obs, ranges_, weights, num_rays, max_particles_);
//     //         for (auto& weight : weights) {
//     //             weight = pow(weight, inv_squash_factor_);
//     //         }
//     //     } else {
//     //         ROS_WARN("Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2");
//     //     }
//     // } else if (rangelib_var_ == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT) {
//     //     for (int i = 0; i < max_particles_; ++i) {
//     //         queries_[i][0] = proposal_dist[i][0];
//     //         queries_[i][1] = proposal_dist[i][1];
//     //         queries_[i][2] = proposal_dist[i][2];
//     //     }
//     //     range_method_->calc_range_repeat_angles_eval_sensor_model(queries_, downsampled_angles_, obs, weights);
//     //     for (auto& weight : weights) {
//     //         weight = pow(weight, inv_squash_factor_);
//     //     }
//     // } else if (rangelib_var_ == VAR_REPEAT_ANGLES_EVAL_SENSOR) {
//     //     if (show_fine_timing_) {
//     //         t_start_ = ros::Time::now();
//     //     }
//     //     for (int i = 0; i < max_particles_; ++i) {
//     //         queries_[i][0] = proposal_dist[i][0];
//     //         queries_[i][1] = proposal_dist[i][1];
//     //         queries_[i][2] = proposal_dist[i][2];
//     //     }
//     //     if (show_fine_timing_) {
//     //         t_init_ = ros::Time::now();
//     //     }
//     //     range_method_->calc_range_repeat_angles(queries_, downsampled_angles_, ranges_);
//     //     if (show_fine_timing_) {
//     //         t_range_ = ros::Time::now();
//     //     }
//     //     range_method_->eval_sensor_model(flat_obs, ranges_, weights, num_rays, max_particles_);
//     //     if (show_fine_timing_) {
//     //         t_eval_ = ros::Time::now();
//     //     }
//     //     for (auto& weight : weights) {
//     //         weight = pow(weight, inv_squash_factor_);
//     //     }
//     //     if (show_fine_timing_) {
//     //         t_squash_ = ros::Time::now();
//     //         t_total_ = (t_squash_ - t_start_).toSec() / 100.0;
//     //         if (iters_ % 10 == 0) {
//     //             ROS_INFO("sensor_model: init: %.2f range: %.2f eval: %.2f squash: %.2f",
//     //                      (t_init_ - t_start_).toSec() / t_total_,
//     //                      (t_range_ - t_init_).toSec() / t_total_,
//     //                      (t_eval_ - t_range_).toSec() / t_total_,
//     //                      (t_squash_ - t_eval_).toSec() / t_total_);
//     //         }
//     //     }
//     // } else if (rangelib_var_ == VAR_CALC_RANGE_MANY_EVAL_SENSOR) {
//     //     for (int i = 0; i < max_particles_; ++i) {
//     //         for (int j = 0; j < num_rays; ++j) {
//     //             queries_[i * num_rays + j][0] = proposal_dist[i][0];
//     //             queries_[i * num_rays + j][1] = proposal_dist[i][1];
//     //             queries_[i * num_rays + j][2] = proposal_dist[i][2] + tiled_angles_[j];
//     //         }
//     //     }

//     //     std::vector<float> flat_queries;
//     //     for (const auto& query : queries_) {
//     //         flat_queries.insert(flat_queries.end(), query.begin(), query.end());
//     //     }

//     //     range_method_->numpy_calc_range(flat_queries.data(), ranges_.data(), ranges_.size());
//     //     range_method_->eval_sensor_model(flat_obs.data(), ranges_.data(), weights.data(), num_rays, max_particles_);
//     //     for (auto& weight : weights) {
//     //         weight = pow(weight, inv_squash_factor_);
//     //     }
//     // } else if (rangelib_var_ == VAR_NO_EVAL_SENSOR_MODEL) {
//     //     for (int i = 0; i < max_particles_; ++i) {
//     //         for (int j = 0; j < num_rays; ++j) {
//     //             queries_[i * num_rays + j][0] = proposal_dist[i][0];
//     //             queries_[i * num_rays + j][1] = proposal_dist[i][1];
//     //             queries_[i * num_rays + j][2] = proposal_dist[i][2] + tiled_angles_[j];
//     //         }
//     //     }
//     //     std::vector<float> flat_queries;
//     //     for (const auto& query : queries_) {
//     //         flat_queries.insert(flat_queries.end(), query.begin(), query.end());
//     //     }

//     //     range_method_->numpy_calc_range(flat_queries.data(), ranges_.data(), ranges_.size());
//     //     std::vector<uint16_t> int_obs(num_rays);
//     //     std::vector<uint16_t> int_rngs(num_rays * max_particles_);

//     //     for (int i = 0; i < num_rays; ++i) {
//     //         int_obs[i] = std::min(static_cast<int>(obs[i] / map_info_.resolution), max_range_px_);
//     //     }
//     //     for (int i = 0; i < num_rays * max_particles_; ++i) {
//     //         int_rngs[i] = std::min(static_cast<int>(ranges_[i] / map_info_.resolution), max_range_px_);
//     //     }

//     //     for (int i = 0; i < max_particles_; ++i) {
//     //         double weight = 1.0;
//     //         for (int j = 0; j < num_rays; ++j) {
//     //             weight *= sensor_model_table_[int_obs[j]][int_rngs[i * num_rays + j]];
//     //         }
//     //         weights[i] = pow(weight, inv_squash_factor_);
//     //     }
//     // } else {
//     //     ROS_WARN("Please set rangelib_variant parameter to 0-4");
//     // }
// // }


// Sensor model function
void ParticleFilter::sensorModel(std::vector<std::vector<double>>& proposal_dist, const std::vector<float>& real_scan, std::vector<double>& weights) {
    int num_rays = downsampled_angles_.size();

    if (first_sensor_update_) {
        if (rangelib_var_ <= 1) {
            queries_ = std::vector<std::vector<float>>(num_rays * max_particles_, std::vector<float>(3, 0.0));
        } else {
            queries_ = std::vector<std::vector<float>>(max_particles_, std::vector<float>(3, 0.0));
        }
        fake_ranges_ = std::vector<float>(num_rays * max_particles_, 0.0);
        tiled_angles_.resize(num_rays * max_particles_);
        for (int i = 0; i < max_particles_; ++i) {
            std::copy(downsampled_angles_.begin(), downsampled_angles_.end(), tiled_angles_.begin() + i * num_rays);
        }
        first_sensor_update_ = false;
    }

    if (rangelib_var_ == VAR_RADIAL_CDDT_OPTIMIZATIONS) {
        if (which_rm_.find("cddt") != std::string::npos) {
            for (int i = 0; i < max_particles_; ++i) {
                queries_[i][0] = proposal_dist[i][0];
                queries_[i][1] = proposal_dist[i][1];
                queries_[i][2] = proposal_dist[i][2];
            }
            std::vector<float> flat_queries;
            for (const auto& query : queries_) {
                flat_queries.insert(flat_queries.end(), query.begin(), query.end());
            }
            range_method_->calc_range_many_radial_optimized(flat_queries.data(), fake_ranges_.data(),
                                                            max_particles_, num_rays, flat_queries.size(), downsampled_angles_.back());
            range_method_->eval_sensor_model(const_cast<float*>(real_scan.data()), fake_ranges_.data(),
                                            weights_.data(), num_rays, max_particles_);
            std::transform(weights_.begin(), weights_.end(), weights_.begin(),
                        [this](double w) { return std::pow(w, inv_squash_factor_); });
        } else {
            RCLCPP_ERROR(this->get_logger(), "Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2");
        }
    } else if (rangelib_var_ == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT) {
        for (int i = 0; i < max_particles_; ++i) {
            queries_[i][0] = proposal_dist[i][0];
            queries_[i][1] = proposal_dist[i][1];
            queries_[i][2] = proposal_dist[i][2];
        }
        std::vector<float> flat_queries;
        for (const auto& query : queries_) {
            flat_queries.insert(flat_queries.end(), query.begin(), query.end());
        }
        range_method_->calc_range_repeat_angles_eval_sensor_model(flat_queries.data(), downsampled_angles_.data(),
                                                                const_cast<float*>(real_scan.data()), weights_.data(),
                                                                max_particles_, downsampled_angles_.size());
        std::transform(weights_.begin(), weights_.end(), weights_.begin(),
                    [this](double w) { return std::pow(w, inv_squash_factor_); });
    } else if (rangelib_var_ == VAR_REPEAT_ANGLES_EVAL_SENSOR) {
        for (int i = 0; i < max_particles_; ++i) {
            queries_[i][0] = proposal_dist[i][0];
            queries_[i][1] = proposal_dist[i][1];
            queries_[i][2] = proposal_dist[i][2];
        }
        std::vector<float> flat_queries;
        for (const auto& query : queries_) {
            flat_queries.insert(flat_queries.end(), query.begin(), query.end());
        }
        range_method_->numpy_calc_range_angles(flat_queries.data(), downsampled_angles_.data(),
                                                fake_ranges_.data(), max_particles_, downsampled_angles_.size());
        range_method_->eval_sensor_model(const_cast<float*>(real_scan.data()), fake_ranges_.data(),
                                        weights_.data(), num_rays, max_particles_);
        std::transform(weights_.begin(), weights_.end(), weights_.begin(),
                    [this](double w) { return std::pow(w, inv_squash_factor_); });
    } else if (rangelib_var_ == VAR_CALC_RANGE_MANY_EVAL_SENSOR) {
        for (int i = 0; i < max_particles_; ++i) {
            for (int j = 0; j < num_rays; ++j) {
                queries_[i * num_rays + j][0] = proposal_dist[i][0];
                queries_[i * num_rays + j][1] = proposal_dist[i][1];
                queries_[i * num_rays + j][2] = proposal_dist[i][2] + downsampled_angles_[j];
            }
        }
        std::vector<float> flat_queries;
        for (const auto& query : queries_) {
            flat_queries.insert(flat_queries.end(), query.begin(), query.end());
        }
        range_method_->numpy_calc_range(flat_queries.data(), fake_ranges_.data(), num_rays);
        range_method_->eval_sensor_model(const_cast<float*>(real_scan.data()), fake_ranges_.data(),
                                        weights_.data(), num_rays, max_particles_);
        std::transform(weights_.begin(), weights_.end(), weights_.begin(),
                    [this](double w) { return std::pow(w, inv_squash_factor_); });
    } else if (rangelib_var_ == VAR_NO_EVAL_SENSOR_MODEL) {
        for (int i = 0; i < max_particles_; ++i) {
            for (int j = 0; j < num_rays; ++j) {
                queries_[i * num_rays + j][0] = proposal_dist[i][0];
                queries_[i * num_rays + j][1] = proposal_dist[i][1];
                queries_[i * num_rays + j][2] = proposal_dist[i][2] + downsampled_angles_[j];
            }
        }
        std::vector<float> flat_queries;
        for (const auto& query : queries_) {
            flat_queries.insert(flat_queries.end(), query.begin(), query.end());
        }
        range_method_->numpy_calc_range(flat_queries.data(), fake_ranges_.data(), num_rays);

        std::vector<uint16_t> intobs(real_scan.size());
        std::vector<uint16_t> intrng(fake_ranges_.size());

        std::transform(real_scan.begin(), real_scan.end(), intobs.begin(),
                    [this](float val) { return static_cast<uint16_t>(std::rint(val / map_info_.resolution)); });
        std::transform(fake_ranges_.begin(), fake_ranges_.end(), intrng.begin(),
                    [this](float val) { return static_cast<uint16_t>(std::rint(val / map_info_.resolution)); });

        for (int i = 0; i < max_particles_; ++i) {
            double weight = 1.0;
            for (int j = 0; j < num_rays; ++j) {
                weight *= sensor_model_table_[intobs[j]][intrng[i * num_rays + j]];
            }
            weight = std::pow(weight, inv_squash_factor_);
            weights_[i] = weight;
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "PLEASE SET rangelib_variant PARAM to 0-4");
    }

}

std::vector<double> ParticleFilter::expectedPose() {
    std::vector<double> expected_pose(3, 0.0); // Initialize a vector of size 3 with all zeros

    for (size_t i = 0; i < particles_.size(); ++i) {
        expected_pose[0] += particles_[i][0] * weights_[i];
        expected_pose[1] += particles_[i][1] * weights_[i];
        expected_pose[2] += particles_[i][2] * weights_[i];
    }

    return expected_pose;
}

void ParticleFilter::update() {
    if (lidar_initialized_ && odom_initialized_ && map_initialized_) {
        // Lock the state with mutex to prevent concurrency issues
        std::unique_lock<std::mutex> lock(state_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            RCLCPP_WARN(this->get_logger(), "Concurrency error avoided");
            return;
        }

        // Timer tick for performance tracking
        timer_.tick();
        iters_ += 1;

        // Record the start time
        auto start_time = this->now();

        // Convert downsampled LiDAR ranges to float vector
        std::vector<float> real_scan(downsampled_ranges_.begin(), downsampled_ranges_.end());

        // Get the initial guess from odometry data
        std::vector<double> initial_guess(odometry_data_.begin(), odometry_data_.end());
        std::fill(odometry_data_.begin(), odometry_data_.end(), 0.0);

        // Run the Monte Carlo Localization (MCL) update
        monteCarloLocalization(initial_guess, real_scan);

        // Compute the expected value of the robot pose
        inferred_pose_ = expectedPose();

        // Record the end time
        auto end_time = this->now();

        // Publish the transformation based on the inferred pose
        publish_tf(inferred_pose_, last_stamp_);

        // Calculate the time taken for this iteration
        auto elapsed_seconds = (end_time - start_time).seconds();
        double ips = 1.0 / elapsed_seconds;
        smoothing_.append(ips);

        // Log performance metrics every 10 iterations
        if (iters_ % 10 == 0) {
            RCLCPP_INFO(this->get_logger(), "Iterations per second: %d, Possible: %d",
                        static_cast<int>(timer_.fps()), static_cast<int>(smoothing_.mean()));
        }

        visualize();
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ParticleFilter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
