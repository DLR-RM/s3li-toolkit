#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/stereo.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <experimental/filesystem>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <image_geometry/pinhole_camera_model.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

std::string strip_last_piece(std::string_view input) {

    // Strip beginning '/' if the string has one
    if (input.at(0) == '/') {
        input.remove_prefix(1);
    }

    // Find the last '/' in the string view
    auto pos = input.find_last_of('/');

    // If we found '/', return the substring before it
    if (pos != std::string_view::npos) {
        return std::string(input.substr(0, pos)); // Return as std::string
    }

    // If no '/' found, return the input as is
    return std::string(input);
}

void writePFM(const cv::Mat &mat, const std::string &filename) {
	// Ensure the matrix is of type CV_32F (float)
	if (mat.type() != CV_32F) {
		std::cerr << "Input matrix is not of type CV_32F!" << std::endl;
		return;
	}

	// Open the file
	std::ofstream file(filename, std::ios::binary);
	if (!file) {
		std::cerr << "Error opening the file!" << std::endl;
		return;
	}

	// Write the header
	file << "Pf\n"; // 'Pf' indicates it's a color float map (single-channel)
	file << mat.cols << " " << mat.rows << "\n"; // Image dimensions
	file << "-1.0\n"; // Byte order (little-endian)

	// Write the pixel data
	for (int y = 0; y < mat.rows; ++y) {
		for (int x = 0; x < mat.cols; ++x) {
			// Write the float values one by one
			float value = mat.at<float>(y, x);
			file.write(reinterpret_cast<char*>(&value), sizeof(float));
		}
	}

	file.close();
}

/**
 *
 * @param img_src image whose values are then to a JET colormap
 * @param img_dst image prepared for display
 */
void add_colorbar(cv::Mat& img_src, cv::Mat& img_dst) {
    // Thanks ChatGPT!
    int colorbarHeight = 30;
    int colorbarWidth = img_dst.cols;
    cv::Mat colorbar(colorbarHeight, colorbarWidth, CV_8UC3);

    // Apply the same colormap to the colorbar (just a gradient)
    cv::Mat gradient(colorbarHeight, colorbarWidth, CV_8UC1);
    for (int i = 0; i < colorbarWidth; ++i) {
        gradient.col(i).setTo(cv::Scalar(i * 255 / colorbarWidth));
    }
    cv::applyColorMap(gradient, colorbar, cv::COLORMAP_JET);

    // Step 4: Overlay the colorbar on the original image (stack vertically)
    colorbar.copyTo(img_dst(cv::Rect(0, 0, colorbar.cols, colorbar.rows)));

    // Step 5: Add text labels at 0%, 25%, 50%, 75%, and 100% values
    // Get the min and max values of the disparity map
    double minVal, maxVal;
    cv::minMaxLoc(img_src, &minVal, &maxVal);

    // Calculate the corresponding values for 25%, 50%, 75%
    double val25 = minVal + 0.25 * (maxVal - minVal);
    double val50 = minVal + 0.50 * (maxVal - minVal);
    double val75 = minVal + 0.75 * (maxVal - minVal);

    std::stringstream ss_min, ss_max, ss_25, ss_50, ss_75;
    ss_min << std::setprecision(2) << std::fixed << minVal;
    ss_25 << std::setprecision(2) << std::fixed << val25;
    ss_50 << std::setprecision(2) << std::fixed << val50;
    ss_75 << std::setprecision(2) << std::fixed << val75;
    ss_max << std::setprecision(2) << std::fixed << maxVal;

    // Define the positions for text on the colorbar (in terms of pixel locations)
    int textY = colorbarHeight / 2; // Vertical position of text on colorbar

    // Add the text labels (min, 25%, 50%, 75%, max)
    cv::putText(img_dst, ss_min.str(), cv::Point(10, textY), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255), 2);
    cv::putText(img_dst, ss_25.str(), cv::Point(colorbarWidth * 0.25, textY), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255), 2);
    cv::putText(img_dst, ss_50.str(), cv::Point(colorbarWidth * 0.50, textY), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255), 2);
    cv::putText(img_dst, ss_75.str(), cv::Point(colorbarWidth * 0.75, textY), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255), 2);
    cv::putText(img_dst, ss_max.str(), cv::Point(colorbarWidth - 40, textY), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255), 2);
}

class StereoImageCapture: public rclcpp::Node
{
public:
    StereoImageCapture();
    static void run();

private:
    void single_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg);
    void callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_img_msg, const sensor_msgs::msg::Image::ConstSharedPtr& right_img_msg);
    void callback_cinfos(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& left_info_msg,
                         const sensor_msgs::msg::CameraInfo::ConstSharedPtr& right_info_msg);
    void compute_rectification_maps(const sensor_msgs::msg::CameraInfo& left_info,
                                    const sensor_msgs::msg::CameraInfo& right_info);

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    
    // Rectification maps
    cv::Mat left_map1_, left_map2_, right_map1_, right_map2_;
    double baseline_ = 0.0;

    sensor_msgs::msg::CameraInfo left_cam_info_rect_, right_cam_info_rect_;

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_img_sub_, right_img_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> left_info_sub_, right_info_sub_;
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::Image>> left_image_single_subscriber_;
    float rescale_disp_;

    // Define the sync policy for approximate time synchronizer
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> ImageApproxSyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::CameraInfo, sensor_msgs::msg::CameraInfo> CInfoApproxSyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<ImageApproxSyncPolicy>> sync_images_;
    std::shared_ptr<message_filters::Synchronizer<CInfoApproxSyncPolicy>> sync_cinfos_;

    // Initialize Stereo params
    cv::Ptr<cv::StereoSGBM> stereo_;

    std::string left_topic_, right_topic_;
    std::string left_info_topic_, right_info_topic_;
    std::string output_dir_ = "/tmp/stereo_disparity_node_dump/";
    
    bool rectify_;
    bool rectification_ready_;
    float rescale_;
    int image_count_;
    bool visualize_;
	bool flip_RT_calde_;
};

StereoImageCapture::StereoImageCapture():
    Node("stereo_depth_node"),
      rectification_ready_(false),
      image_count_(0)
{
    this->declare_parameter<std::string>("left_image_topic", "/left/image_raw");
    this->declare_parameter<std::string>("right_image_topic", "/right/image_raw");
    this->declare_parameter<std::string>("left_camera_info_topic", "/left/camera_info");
    this->declare_parameter<std::string>("right_camera_info_topic", "/right/camera_info");
    this->declare_parameter<std::string>("output_dir", "/tmp/stereo_images");
    this->declare_parameter<bool>("rectify", true);
    this->declare_parameter<bool>("visualize", false);
    this->declare_parameter<float>("rescale", 1.0);
    this->declare_parameter<float>("rescale_disp", 0.0);
    this->declare_parameter<bool>("flip_RT_calde", true);

    // Retrieve parameter values into member variables
    left_topic_       = this->get_parameter("left_image_topic").as_string();
    right_topic_      = this->get_parameter("right_image_topic").as_string();
    left_info_topic_  = this->get_parameter("left_camera_info_topic").as_string();
    right_info_topic_ = this->get_parameter("right_camera_info_topic").as_string();
    output_dir_       = this->get_parameter("output_dir").as_string();
    rectify_          = this->get_parameter("rectify").as_bool();
    visualize_        = this->get_parameter("visualize").as_bool();
    rescale_          = this->get_parameter("rescale").as_double();
    rescale_disp_     = this->get_parameter("rescale_disp").as_double();
    flip_RT_calde_    = this->get_parameter("flip_RT_calde").as_bool();

    // Create output directory if not exists
    if (!std::experimental::filesystem::exists(output_dir_)) {
        std::experimental::filesystem::create_directories(output_dir_);
    }

    // Set up publishers
    // Remove trailing parts and append suffixes, just like strip_last_piece() did
    std::string left_rect_topic = strip_last_piece(left_topic_) + "/image_rect";
    std::string right_rect_topic = strip_last_piece(right_topic_) + "/image_rect";
    std::string left_info_topic = strip_last_piece(left_topic_) + "/image_rect/camera_info";
    std::string right_info_topic = strip_last_piece(right_topic_) + "/image_rect/camera_info";

    // Create publishers
    left_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(left_rect_topic, 1);
    right_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(right_rect_topic, 1);
    left_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(left_info_topic, 1);
    right_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(right_info_topic, 1);

    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("depth", 1);
    pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pointcloud", 5);

    // Stereo SGBM configuration
    int block_size = 7;
    stereo_ = cv::StereoSGBM::create(
        0, 64, block_size,
        8 * 1 * block_size * block_size,
        32 * 1 * block_size * block_size,
        1, 10, 60,
        2, 63,
        cv::StereoSGBM::MODE_HH
    );

    RCLCPP_INFO(this->get_logger(), "StereoImageCapture initialized with rescale factor %.2f", rescale_);
    if (visualize_) {
        cv::namedWindow("Stereo View", cv::WINDOW_NORMAL);
        RCLCPP_INFO(this->get_logger(), "Press 's' in the image window to save synchronized images.");
    }

    // Regular (non-filtered) subscriber for single images
    /*
    left_image_single_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
        left_topic_, 10,
        std::bind(&StereoImageCapture::single_image_callback, this, std::placeholders::_1)
    );
    */

    // Setup filtered subscribers for synchronized stereo images
    left_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, left_topic_);
    right_img_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, right_topic_);

    sync_images_ = std::make_shared<message_filters::Synchronizer<ImageApproxSyncPolicy>>(
        ImageApproxSyncPolicy(10), *left_img_sub_, *right_img_sub_);
    sync_images_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.01));

    sync_images_->registerCallback(
        std::bind(&StereoImageCapture::callback, this, std::placeholders::_1, std::placeholders::_2)
    );

    // Setup filtered subscribers for synchronized CameraInfo
    left_info_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(this, left_info_topic_);
    right_info_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(this, right_info_topic_);

    sync_cinfos_ = std::make_shared<message_filters::Synchronizer<CInfoApproxSyncPolicy>>(
        CInfoApproxSyncPolicy(10), *left_info_sub_, *right_info_sub_);

    sync_cinfos_->registerCallback(
        std::bind(&StereoImageCapture::callback_cinfos, this, std::placeholders::_1, std::placeholders::_2)
    );
}


void StereoImageCapture::callback_cinfos(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& left_info_msg,
                                         const sensor_msgs::msg::CameraInfo::ConstSharedPtr& right_info_msg) {
    if (!rectification_ready_) {
        compute_rectification_maps(*left_info_msg, *right_info_msg);
    }
}

void StereoImageCapture::single_image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg) {
    if (!rectification_ready_) {
        RCLCPP_WARN(this->get_logger(), "Waiting for rect info to be ready...");
        return;
    }

    // Convert to OpenCV images
    cv_bridge::CvImagePtr cv_img;
    try {
        cv_img = cv_bridge::toCvCopy(img_msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat img = cv_img->image;

    // Rectify
    cv::Mat img_rect;
    
    if (rectify_) {
        cv::remap(img, img_rect, left_map1_, left_map2_, cv::INTER_LINEAR);
    } else {
        img_rect = img; 
    }
    
     // Resize if needed
    if (rescale_ != 1.0) {
        cv::resize(img_rect, img_rect, cv::Size(), rescale_, rescale_);
    }

    // Publish image
    cv_bridge::CvImage msg;
    msg.header = img_msg->header;
    msg.encoding = "bgr8";
    msg.image = img_rect;
    left_img_pub_->publish(*msg.toImageMsg());

}


void StereoImageCapture::callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_img_msg,
                                  const sensor_msgs::msg::Image::ConstSharedPtr& right_img_msg) {
    if (!rectification_ready_) {
        RCLCPP_WARN(this->get_logger(), "Waiting for rect info to be ready...");
        return;
    }

    // Convert to OpenCV images
    cv_bridge::CvImagePtr cv_left, cv_right;
    try {
        cv_left = cv_bridge::toCvCopy(left_img_msg, "bgr8");
        cv_right = cv_bridge::toCvCopy(right_img_msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat left_img = cv_left->image;
    cv::Mat right_img = cv_right->image;

    // Rectify
    cv::Mat left_rect, right_rect;
    if (rectify_) {
        cv::remap(left_img, left_rect, left_map1_, left_map2_, cv::INTER_LINEAR);
        cv::remap(right_img, right_rect, right_map1_, right_map2_, cv::INTER_LINEAR);
    } else {
        left_rect = left_img; 
        right_rect = right_img; 
    }
    
    // Resize if needed
    if (rescale_ != 1.0) {
        cv::resize(left_rect, left_rect, cv::Size(), rescale_, rescale_);
        cv::resize(right_rect, right_rect, cv::Size(), rescale_, rescale_);
    }

    // Convert to grayscale
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_rect, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_rect, right_gray, cv::COLOR_BGR2GRAY);

    // Compute disparity
    cv::Mat disparity_16bit, disparity, depth_map;
    
    if (rectify_) {
        stereo_->compute(left_gray, right_gray, disparity_16bit);

        disparity_16bit.convertTo(disparity, CV_32F);

        disparity *= 1.0 / (16.0);
        disparity.setTo(NAN, disparity <= 0.0);

        // Compute depth map (placeholder - fill in with real intrinsics later)
        depth_map = cv::Mat::zeros(disparity.size(), CV_32F);
        depth_map = baseline_ * left_cam_info_rect_.p[0] / (rescale_disp_ + disparity);

        // Publish depth image
        cv_bridge::CvImage depth_cv;
        depth_cv.header = left_img_msg->header;
        depth_cv.encoding = "32FC1";
        depth_cv.image = depth_map;
        depth_pub_->publish(*depth_cv.toImageMsg());
        
        // Publish left image
        cv_bridge::CvImage left_image_rect_msg;
        left_image_rect_msg.header = left_img_msg->header;
        left_image_rect_msg.encoding = "bgr8";
        left_image_rect_msg.image = left_rect;
        left_img_pub_->publish(*left_image_rect_msg.toImageMsg());

        // Publish right image
        cv_bridge::CvImage right_image_rect_msg;
        right_image_rect_msg.header = right_img_msg->header;
        right_image_rect_msg.encoding = "bgr8";
        right_image_rect_msg.image = right_rect;
        right_img_pub_->publish(*right_image_rect_msg.toImageMsg());

        // Publish camera infos
        left_cam_info_rect_.header = left_img_msg->header;
        right_cam_info_rect_.header = right_img_msg->header;
        left_info_pub_->publish(left_cam_info_rect_);
        right_info_pub_->publish(right_cam_info_rect_);

        // Use the camera model to convert depth to 3D points
        image_geometry::PinholeCameraModel model;
        model.fromCameraInfo(left_cam_info_rect_);

        // Iterate over the depth image
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        for (int row = 0; row < depth_cv.image.rows; ++row) {
            for (int col = 0; col < depth_cv.image.cols; ++col) {
                float depth = depth_cv.image.at<float>(row, col);
                if (depth == 0 || isnan(depth)) {
                    continue;  // Skip points with no depth data (zero depth)
                }

                // Convert pixel to 3D point
                auto point = model.projectPixelTo3dRay(cv::Point2d(col, row));
                point.x *= depth;
                point.y *= depth;
                point.z *= depth;
                pcl::PointXYZRGB point3d;
                point3d.x = point.x;
                point3d.y = point.y;
                point3d.z = point.z;
                point3d.r = left_rect.at<cv::Vec3b>(row, col)[2];
                point3d.g = left_rect.at<cv::Vec3b>(row, col)[1];
                point3d.b = left_rect.at<cv::Vec3b>(row, col)[0];

                cloud.push_back(point3d);
            }
        }

        // Publish the point cloud
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(cloud, cloud_msg);
        cloud_msg.header = left_img_msg->header;
        pc_pub_->publish(cloud_msg);
    }

    // Visualization
    if (visualize_) {
        cv::Mat disp_vis, disp_depth;
        if (rectify_) {
            cv::normalize(disparity, disp_vis, 0, 255, cv::NORM_MINMAX);
            disp_vis.convertTo(disp_vis, CV_8U);
            cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);

            cv::normalize(depth_map, disp_depth, 0, 255, cv::NORM_MINMAX);
            disp_depth.convertTo(disp_depth, CV_8U);
            cv::applyColorMap(disp_depth, disp_depth, cv::COLORMAP_JET);

            add_colorbar(disparity_16bit, disp_vis);
            add_colorbar(depth_map, disp_depth);
        }
        
        cv::Mat vis, vis_1row, vis_2row;
        cv::hconcat(left_rect, right_rect, vis_1row);
        
        if (rectify_) {
            cv::hconcat(disp_vis, disp_depth, vis_2row);
			cv::vconcat(vis_1row, vis_2row, vis);
        } else {
        	vis = vis_1row;
        }

        if (vis.rows > 800) {
            double rescale_vis = 800.0 / static_cast<double>(vis.rows);
            cv::resize(vis, vis, cv::Size(), rescale_vis, rescale_vis);
        }

        for (int y = 0; y < vis.rows; y += 20) {
            cv::line(vis, cv::Point(0, y), cv::Point(vis.cols, y), cv::Scalar(0, 255, 0), 1);
        }

		vis(cv::Rect(0, 0, vis.cols, 20)).setTo(0);
		cv::putText(vis, "t_left: " + std::to_string(rclcpp::Time(left_img_msg->header.stamp).seconds()), cv::Point(10, 12),
			cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1);
		cv::putText(vis, "t_right: " + std::to_string(rclcpp::Time(right_img_msg->header.stamp).seconds()), cv::Point(vis.cols / 2 + 10, 12),
					cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1);
		cv::putText(vis, "dt: " + std::to_string(std::fabs(rclcpp::Time(right_img_msg->header.stamp).seconds() - rclcpp::Time(left_img_msg->header.stamp).seconds())),
					cv::Point(0.4f * float(vis.cols), 12),
					cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1);

        cv::imshow("Stereo View", vis);
        char key = static_cast<char>(cv::waitKey(10));
        if (key == 's') {
            std::string base = output_dir_ + "/shot" + std::to_string(image_count_);
            cv::imwrite(base + ".left.png", left_img);
            cv::imwrite(base + ".right.png", right_img);
            if (rectify_) {
                cv::imwrite(base + ".rect_left.png", left_rect);
                cv::imwrite(base + ".rect_right.png", right_rect);
                writePFM(disparity, base + ".disp_32bit.pfm" );
                writePFM(depth_map, base + ".depth_32bit.pfm");
            }
            image_count_++;
            RCLCPP_INFO(this->get_logger(), "Saved stereo image %d", image_count_);
        }
    }
}

void StereoImageCapture::compute_rectification_maps(const sensor_msgs::msg::CameraInfo& left_info,
                                                    const sensor_msgs::msg::CameraInfo& right_info)
{
    // Convert camera info to cv::Mat
    cv::Mat K1 = cv::Mat(3, 3, CV_64F, const_cast<double*>(left_info.k.data())).clone();
    cv::Mat D1 = cv::Mat(left_info.d).clone();

    cv::Mat K2 = cv::Mat(3, 3, CV_64F, const_cast<double*>(right_info.k.data())).clone();
    cv::Mat D2 = cv::Mat(right_info.d).clone();

    cv::Mat R = cv::Mat(3, 3, CV_64F, const_cast<double*>(right_info.r.data())).clone();

    cv::Mat T = cv::Mat::zeros(3, 1, CV_64F);
    T.at<double>(0, 0) = right_info.p[3];
    T.at<double>(0, 1) = right_info.p[7];
    T.at<double>(0, 2) = right_info.p[11];

    // Compute rectification maps
    cv::Size size(left_info.width, left_info.height);

    cv::Mat P1, P2, R1, R2, Q;

	if (!flip_RT_calde_) {
		cv::stereoRectify(K1, D1, K2, D2,
						  size, R, T, R1, R2, P1, P2, Q,
						  cv::CALIB_ZERO_DISPARITY, 1);
	} else {
		cv::stereoRectify(K1, D1, K2, D2,
						  size, R.inv(), -R.inv()*T, R1, R2, P1, P2, Q);
	}

    cv::initUndistortRectifyMap(K1, D1, R1, P1, size,
                                CV_16SC2, left_map1_, left_map2_);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, size,
                                CV_16SC2, right_map1_, right_map2_);

    baseline_ = std::fabs(P2.at<double>(0, 3) / P2.at<double>(0, 0));

    rectification_ready_ = true;
    RCLCPP_INFO(this->get_logger(), "Computed rectification maps. Baseline: %.4f meters", baseline_);

    if (rescale_ != 1.0) {
        size = cv::Size(left_info.width * rescale_, left_info.height * rescale_);
        cv::Mat scale = (cv::Mat_<double>(3, 3) << rescale_, 0, 0,
                                                   0, rescale_, 0,
                                                   0, 0, 1);
        P1(cv::Rect(0, 0, 3, 3)) = scale * P1(cv::Rect(0, 0, 3, 3));
        P2(cv::Rect(0, 0, 3, 3)) = scale * P2(cv::Rect(0, 0, 3, 3));
        P2.at<double>(0, 3) *= rescale_;
    }

    // Save rectified camera info
    left_cam_info_rect_ = sensor_msgs::msg::CameraInfo();
    cv::Mat K1_rect, K2_rect;
    P1(cv::Rect(0, 0, 3, 3)).copyTo(K1_rect);
    std::copy(K1_rect.begin<double>(), K1_rect.end<double>(), left_cam_info_rect_.k.begin());
    std::copy(P1.begin<double>(), P1.end<double>(), left_cam_info_rect_.p.begin());
    left_cam_info_rect_.width = size.width;
    left_cam_info_rect_.height = size.height;

    right_cam_info_rect_ = sensor_msgs::msg::CameraInfo();
    P2(cv::Rect(0, 0, 3, 3)).copyTo(K2_rect);
    std::copy(K2_rect.begin<double>(), K2_rect.end<double>(), right_cam_info_rect_.k.begin());
    std::copy(P2.begin<double>(), P2.end<double>(), right_cam_info_rect_.p.begin());
    right_cam_info_rect_.width = size.width;
    right_cam_info_rect_.height = size.height;
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<StereoImageCapture>();

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();

    return 0;
}
