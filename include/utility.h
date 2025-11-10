#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/imgproc.hpp>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/search/impl/search.hpp>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER };

class ParamServer {
 public:
	ros::NodeHandle nh;

	std::string robot_id;

	// Topics
	string pointCloudTopic;
	string imuTopic;
	string odomTopic;
	string gpsTopic;

	// Frames
	string lidarFrame;
	string baselinkFrame;
	string odometryFrame;
	string mapFrame;

	// GPS Settings
	bool useImuHeadingInitialization;
	bool useGpsElevation;
	float gpsCovThreshold;
	float poseCovThreshold;

	// Save pcd
	bool savePCD;
	string savePCDDirectory;
    string save_traj_to_path;

	// Lidar Sensor Configuration
	SensorType sensor;
	int N_SCAN;
	int Horizon_SCAN;
	int downsampleRate;
	float lidarMinRange;
	float lidarMaxRange;

	// IMU
	float imuAccNoise;
	float imuGyrNoise;
	float imuAccBiasN;
	float imuGyrBiasN;
	float imuGravity;
	float imuRPYWeight;
	vector<double> extRotV;
    vector<double> extRPYV;
	vector<double> extTransV;
	Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
	Eigen::Vector3d extTrans;
	Eigen::Quaterniond extQRPY;

	// LOAM
	float edgeThreshold;
	float surfThreshold;
	int edgeFeatureMinValidNum;
	int surfFeatureMinValidNum;

	// voxel filter paprams
	float odometrySurfLeafSize;
	float mappingCornerLeafSize;
	float mappingSurfLeafSize;

	float z_tollerance;
	float rotation_tollerance;

	// CPU Params
	int numberOfCores;
	double mappingProcessInterval;

	// Surrounding map
	float surroundingkeyframeAddingDistThreshold;
	float surroundingkeyframeAddingAngleThreshold;
	float surroundingKeyframeDensity;
	float surroundingKeyframeSearchRadius;

	// Loop closure
	bool loopClosureEnableFlag;
	float loopClosureFrequency;
	int surroundingKeyframeSize;
	float historyKeyframeSearchRadius;
	float historyKeyframeSearchTimeDiff;
	int historyKeyframeSearchNum;
	float historyKeyframeFitnessScore;

	// global map visualization radius
	float globalMapVisualizationSearchRadius;
	float globalMapVisualizationPoseDensity;
	float globalMapVisualizationLeafSize;

	// Dynamic object data association
	float staticDetectionMatchThreshold;
    float dynamicDetectionMatchThreshold;
	vector<double> dataAssociationVarianceVector;
    vector<double> objectCloudPriorVariance;
    vector<double> objectCloudOdomVariance;
	vector<double> earlyLooselyCoupledMatchingVarianceVector;
	vector<double> looselyCoupledMatchingVarianceVector;
	vector<double> tightlyCoupledMatchingVarianceVector;

	Eigen::Matrix<double, 6, 1> dataAssociationVarianceEigenVector;
    Eigen::Matrix<double, 6, 1> objectCloudPriorVarianceEigenVector;
    Eigen::Matrix<double, 6, 1> objectCloudOdomVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> earlyLooselyCoupledMatchingVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> looselyCoupledMatchingVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> tightlyCoupledMatchingVarianceEigenVector;

	// Factor covariance matrices (presented as diagonal vectors)
	vector<double> priorOdometryDiagonalVarianceVector;
	vector<double> odometryDiagonalVarianceVector;
	vector<double> earlyConstantVelocityDiagonalVarianceVector;
	vector<double> constantVelocityDiagonalVarianceVector;
	vector<double> motionDiagonalVarianceVector;
	vector<double> looselyCoupledDetectionVarianceVector;
	vector<double> tightlyCoupledDetectionVarianceVector;

	Eigen::Matrix<double, 6, 1> priorOdometryDiagonalVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> odometryDiagonalVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> earlyConstantVelocityDiagonalVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> constantVelocityDiagonalVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> motionDiagonalVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> looselyCoupledDetectionVarianceEigenVector;
	Eigen::Matrix<double, 6, 1> tightlyCoupledDetectionVarianceEigenVector;

	// Gentle coupling options
	int numberOfEarlySteps;
	int numberOfPreLooseCouplingSteps;
	int numberOfVelocityConsistencySteps;
	int numberOfInterLooseCouplingSteps;

	float objectAngularVelocityConsistencyVarianceThreshold;
	float objectLinearVelocityConsistencyVarianceThreshold;

	// Tracking
	int trackingStepsForLostObject;

	ParamServer() {
        // bool ros::NodeHandle::param<T>(const std::string & param_name, T & param_val, const T & default_val) const
        // T: 模板参数，用来从参数服务器中获取的参数类型
        // param_name: 表示参数的名称。需要提供一个字符串来指定要在参数服务器中查询的参数的名称。
        // param_val：变量，从参数服务器中获取的名称为 param_name 的参数值将被存储到该变量中。这个变量的类型必须与模板参数 <T> 匹配。
        // default_val: 默认参数值。如果在参数服务器中没有找到名为 param_name 的参数，那么 param_val 变量将被设置为这个默认值 default_val
		nh.param<std::string>("/robot_id", robot_id, "roboat"); // 斜杠 / 意味着这是一个全局参数名，无论哪个节点都可以通过这个全路径来访问该参数

		nh.param<std::string>("lio_track/pointCloudTopic", pointCloudTopic, "points_raw");
		nh.param<std::string>("lio_track/imuTopic", imuTopic, "imu_correct");
		nh.param<std::string>("lio_track/odomTopic", odomTopic, "odometry/imu");
		nh.param<std::string>("lio_track/gpsTopic", gpsTopic, "odometry/gps");

		nh.param<std::string>("lio_track/lidarFrame", lidarFrame, "base_link");
		nh.param<std::string>("lio_track/baselinkFrame", baselinkFrame, "base_link");
		nh.param<std::string>("lio_track/odometryFrame", odometryFrame, "odom");
		nh.param<std::string>("lio_track/mapFrame", mapFrame, "map");

		nh.param<bool>("lio_track/useImuHeadingInitialization", useImuHeadingInitialization, false);
		nh.param<bool>("lio_track/useGpsElevation", useGpsElevation, false);
		nh.param<float>("lio_track/gpsCovThreshold", gpsCovThreshold, 2.0);
		nh.param<float>("lio_track/poseCovThreshold", poseCovThreshold, 25.0);

		nh.param<bool>("lio_track/savePCD", savePCD, false);
		nh.param<std::string>("lio_track/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");
        nh.param<std::string>("lio_track/save_traj_to_path", save_traj_to_path, "/media/yzc/MyPassport/slam_dataset/odometry/01/pred_traj.txt");

		std::string sensorStr;
		nh.param<std::string>("lio_track/sensor", sensorStr, "");
		if (sensorStr == "velodyne") {
			sensor = SensorType::VELODYNE;
		} else if (sensorStr == "ouster") {
			sensor = SensorType::OUSTER;
		} else {
			ROS_ERROR_STREAM(
					"Invalid sensor type (must be either 'velodyne' or 'ouster'): " << sensorStr);
			ros::shutdown();
		}

		nh.param<int>("lio_track/N_SCAN", N_SCAN, 16);
		nh.param<int>("lio_track/Horizon_SCAN", Horizon_SCAN, 1800);
		nh.param<int>("lio_track/downsampleRate", downsampleRate, 1);
		nh.param<float>("lio_track/lidarMinRange", lidarMinRange, 1.0);
		nh.param<float>("lio_track/lidarMaxRange", lidarMaxRange, 1000.0);

		nh.param<float>("lio_track/imuAccNoise", imuAccNoise, 0.01);
		nh.param<float>("lio_track/imuGyrNoise", imuGyrNoise, 0.001);
		nh.param<float>("lio_track/imuAccBiasN", imuAccBiasN, 0.0002);
		nh.param<float>("lio_track/imuGyrBiasN", imuGyrBiasN, 0.00003);
		nh.param<float>("lio_track/imuGravity", imuGravity, 9.80511);
		nh.param<float>("lio_track/imuRPYWeight", imuRPYWeight, 0.01);
		nh.param<vector<double>>("lio_track/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>("lio_track/extrinsicRPY", extRPYV, vector<double>());
		nh.param<vector<double>>("lio_track/extrinsicTrans", extTransV, vector<double>());
		extRot   = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3); // R_{li}
        extRPY   = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3); // q_{li}
		extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1); // t^{l}_{li}
        extQRPY  = Eigen::Quaterniond(extRPY).inverse();

		nh.param<float>("lio_track/edgeThreshold", edgeThreshold, 0.12);
		nh.param<float>("lio_track/surfThreshold", surfThreshold, 0.1);
		nh.param<int>("lio_track/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
		nh.param<int>("lio_track/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

		nh.param<float>("lio_track/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
		nh.param<float>("lio_track/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
		nh.param<float>("lio_track/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

		nh.param<float>("lio_track/z_tollerance", z_tollerance, FLT_MAX);
		nh.param<float>("lio_track/rotation_tollerance", rotation_tollerance, FLT_MAX);

		nh.param<int>("lio_track/numberOfCores", numberOfCores, 2);
		nh.param<double>("lio_track/mappingProcessInterval", mappingProcessInterval, 0.15);

		nh.param<float>("lio_track/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
		nh.param<float>("lio_track/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
		nh.param<float>("lio_track/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
		nh.param<float>("lio_track/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

		nh.param<bool>("lio_track/loopClosureEnableFlag", loopClosureEnableFlag, false);
		nh.param<float>("lio_track/loopClosureFrequency", loopClosureFrequency, 1.0);
		nh.param<int>("lio_track/surroundingKeyframeSize", surroundingKeyframeSize, 50);
		nh.param<float>("lio_track/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
		nh.param<float>("lio_track/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
		nh.param<int>("lio_track/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
		nh.param<float>("lio_track/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

		nh.param<float>("lio_track/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
		nh.param<float>("lio_track/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
		nh.param<float>("lio_track/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

		nh.param<float>("lio_track/staticDetectionMatchThreshold", staticDetectionMatchThreshold, 1.2);
        nh.param<float>("lio_track/dynamicDetectionMatchThreshold", dynamicDetectionMatchThreshold, 19.5);
		nh.param<vector<double>>("lio_track/dataAssociationVarianceVector", dataAssociationVarianceVector, {1e-4, 1e-4, 1e-4, 1e-2, 2e-3, 2e-3});
        nh.param<std::vector<double>>("lio_track/objectCloudPriorVariance", objectCloudPriorVariance, {1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1});
        nh.param<std::vector<double>>("lio_track/objectCloudOdomVariance", objectCloudOdomVariance, {1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2});
		nh.param<vector<double>>("lio_track/earlyLooselyCoupledMatchingVarianceVector", earlyLooselyCoupledMatchingVarianceVector, {1e-4, 1e-4, 1e-4, 1e-2, 2e-3, 2e-3});
		nh.param<vector<double>>("lio_track/looselyCoupledMatchingVarianceVector", looselyCoupledMatchingVarianceVector, {1e-4, 1e-4, 1e-4, 1e-2, 2e-3, 2e-3});
		nh.param<vector<double>>("lio_track/tightlyCoupledMatchingVarianceVector", tightlyCoupledMatchingVarianceVector, {1e-4, 1e-4, 1e-4, 1e-2, 2e-3, 2e-3});

		nh.param<vector<double>>("lio_track/priorOdometryDiagonalVarianceVector", priorOdometryDiagonalVarianceVector, {1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8});
		nh.param<vector<double>>("lio_track/odometryDiagonalVarianceVector", odometryDiagonalVarianceVector, {1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4});
		nh.param<vector<double>>("lio_track/earlyConstantVelocityDiagonalVarianceVector", earlyConstantVelocityDiagonalVarianceVector, {1e-3, 1e-3, 1e-3, 2e-1, 1e-1, 1e-1});
		nh.param<vector<double>>("lio_track/constantVelocityDiagonalVarianceVector", constantVelocityDiagonalVarianceVector, {1e-3, 1e-3, 1e-3, 2e-1, 1e-1, 1e-1});
		nh.param<vector<double>>("lio_track/motionDiagonalVarianceVector", motionDiagonalVarianceVector, {1e-4, 1e-4, 1e-2, 1e-1, 1e-2, 1e-2});
		nh.param<vector<double>>("lio_track/looselyCoupledDetectionVarianceVector", looselyCoupledDetectionVarianceVector, {1e-4, 1e-4, 1e-4, 1e-2, 2e-3, 2e-3});
		nh.param<vector<double>>("lio_track/tightlyCoupledDetectionVarianceVector", tightlyCoupledDetectionVarianceVector, {1e-4, 1e-4, 1e-4, 1e-2, 2e-3, 2e-3});

		
        dataAssociationVarianceEigenVector             = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(dataAssociationVarianceVector.data());
		objectCloudPriorVarianceEigenVector            = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(objectCloudPriorVariance.data());
        objectCloudOdomVarianceEigenVector             = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(objectCloudOdomVariance.data());
        earlyLooselyCoupledMatchingVarianceEigenVector = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(earlyLooselyCoupledMatchingVarianceVector.data());
		looselyCoupledMatchingVarianceEigenVector      = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(looselyCoupledMatchingVarianceVector.data());
		tightlyCoupledMatchingVarianceEigenVector      = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(tightlyCoupledMatchingVarianceVector.data());

		priorOdometryDiagonalVarianceEigenVector         = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(priorOdometryDiagonalVarianceVector.data());
		odometryDiagonalVarianceEigenVector              = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(odometryDiagonalVarianceVector.data());
		earlyConstantVelocityDiagonalVarianceEigenVector = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(earlyConstantVelocityDiagonalVarianceVector.data());
		constantVelocityDiagonalVarianceEigenVector      = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(constantVelocityDiagonalVarianceVector.data());
		motionDiagonalVarianceEigenVector                = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(motionDiagonalVarianceVector.data());
		looselyCoupledDetectionVarianceEigenVector       = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(looselyCoupledDetectionVarianceVector.data());
		tightlyCoupledDetectionVarianceEigenVector       = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(tightlyCoupledDetectionVarianceVector.data());

		nh.param<int>("lio_track/numberOfEarlySteps", numberOfEarlySteps, 1);
		nh.param<int>("lio_track/numberOfPreLooseCouplingSteps", numberOfPreLooseCouplingSteps, 5);
		nh.param<int>("lio_track/numberOfVelocityConsistencySteps", numberOfVelocityConsistencySteps, 3);
		nh.param<int>("lio_track/numberOfInterLooseCouplingSteps", numberOfInterLooseCouplingSteps, 0);

		nh.param<float>("lio_track/objectAngularVelocityConsistencyVarianceThreshold", objectAngularVelocityConsistencyVarianceThreshold, 1e-5);
		nh.param<float>("lio_track/objectLinearVelocityConsistencyVarianceThreshold", objectLinearVelocityConsistencyVarianceThreshold, 1e-2);

		nh.param<int>("lio_track/trackingStepsForLostObject", trackingStepsForLostObject, 3);

		usleep(100);
	}

	sensor_msgs::Imu imuConverter(const sensor_msgs::Imu &imu_in) {
		sensor_msgs::Imu imu_out = imu_in;
		// rotate acceleration
		Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
		acc                           = extRot * acc; // R_{li}*acc_{i} --> acc_{l} 
		imu_out.linear_acceleration.x = acc.x();
		imu_out.linear_acceleration.y = acc.y();
		imu_out.linear_acceleration.z = acc.z();
		// rotate gyroscope
		Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
		gyr                        = extRot * gyr; // R_{li}*gyr_{i} --> gyr_{l} 
		imu_out.angular_velocity.x = gyr.x();
		imu_out.angular_velocity.y = gyr.y();
		imu_out.angular_velocity.z = gyr.z();
		// rotate roll pitch yaw
		Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
		Eigen::Quaterniond q_final = q_from * extQRPY; // q_{earth,i}*q_{li}^{-1} --> q_{earth,l}
		imu_out.orientation.x      = q_final.x();
		imu_out.orientation.y      = q_final.y();
		imu_out.orientation.z      = q_final.z();
		imu_out.orientation.w      = q_final.w();

		if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() + q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1) {
			ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
			ros::shutdown();
		}

		return imu_out;
	}
};

sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame) {
	sensor_msgs::PointCloud2 tempCloud;
	pcl::toROSMsg(*thisCloud, tempCloud); // 将 pcl::PointCloud<PointType>::Ptr 类型的 thisCloud 转变为 ROS 消息 tempCloud
	tempCloud.header.stamp    = thisStamp;
	tempCloud.header.frame_id = thisFrame;
	if (thisPub->getNumSubscribers() != 0)
		thisPub->publish(tempCloud);
	return tempCloud;
}

template <typename T>
double ROS_TIME(T msg) {
	return msg->header.stamp.toSec();
}

template <typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z) {
	*angular_x = thisImuMsg->angular_velocity.x;
	*angular_y = thisImuMsg->angular_velocity.y;
	*angular_z = thisImuMsg->angular_velocity.z;
}

template <typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z) {
	*acc_x = thisImuMsg->linear_acceleration.x;
	*acc_y = thisImuMsg->linear_acceleration.y;
	*acc_z = thisImuMsg->linear_acceleration.z;
}

template <typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw) {
	double imuRoll, imuPitch, imuYaw;
	tf::Quaternion orientation;
	tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
	tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

	*rosRoll  = imuRoll;
	*rosPitch = imuPitch;
	*rosYaw   = imuYaw;
}

float pointDistance(PointType p) {
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

float pointDistance(PointType p1, PointType p2) {
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

#endif;
