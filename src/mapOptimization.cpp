#include <jsk_topic_tools/color_utils.h>
#include "factor.h"
#include "lio_track/Diagnosis.h"
#include "lio_track/ObjectStateArray.h"
#include "lio_track/cloud_info.h"
#include "lio_track/detection.h"
#include "lio_track/flags.h"
#include "lio_track/save_estimation_result.h"
#include "lio_track/save_map.h"
#include "solver.h"
#include "utility.h"

#include <visualization_msgs/MarkerArray.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

// #define ENABLE_COMPACT_VERSION_OF_FACTOR_GRAPH
// #define MAP_OPTIMIZATION_DEBUG
#define ENABLE_SIMULTANEOUS_LOCALIZATION_AND_TRACKING
#define ENABLE_ASYNCHRONOUS_STATE_ESTIMATE_FOR_SLOT
// #define ENABLE_MINIMAL_MEMORY_USAGE

using namespace gtsam;

using symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G;  // GPS pose
using symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

using BoundingBox              = jsk_recognition_msgs::BoundingBox;
using BoundingBoxPtr           = jsk_recognition_msgs::BoundingBoxPtr;
using BoundingBoxConstPtr      = jsk_recognition_msgs::BoundingBoxConstPtr;
using BoundingBoxArray         = jsk_recognition_msgs::BoundingBoxArray;
using BoundingBoxArrayPtr      = jsk_recognition_msgs::BoundingBoxArrayPtr;
using BoundingBoxArrayConstPtr = jsk_recognition_msgs::BoundingBoxArrayConstPtr;

struct PointXYZIRPYT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;  // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;                   // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))
typedef PointXYZIRPYT PointTypePose;

class ObjectStateESKF {
 public:
	Pose3 pose                      = Pose3::identity();
	Pose3 velocity                  = Pose3::identity();
	uint64_t poseNodeIndex          = 0;
	uint64_t velocityNodeIndex      = 0;
	uint64_t objectIndex            = 0;
	uint64_t objectIndexForTracking = 0;
	int lostCount                   = 0;
	int trackScore                  = 0;
	ros::Time timestamp             = ros::Time();

	BoundingBox box       = BoundingBox();
	BoundingBox detection = BoundingBox();

	bool isTightlyCoupled = false;
	bool isFirst          = false;

	TightlyCoupledDetectionFactor::shared_ptr tightlyCoupledDetectionFactorPtr = nullptr;
	LooselyCoupledDetectionFactor::shared_ptr looselyCoupledDetectionFactorPtr = nullptr;
	StablePoseFactor::shared_ptr motionFactorPtr                               = nullptr;
    

	std::vector<uint64_t> previousVelocityNodeIndices;
    // ESKF
    using Mat3T = Eigen::Matrix<double,3,3>;
    using Mat6T = Eigen::Matrix<double,6,6>;
    using Vec6T = Eigen::Matrix<double,6,1>;

    Mat6T poseCovariance = Mat6T::Identity(); // 位姿协方差矩阵
    Mat6T propagationNoise; // 递推噪声协方差矩阵
    Mat6T observationNoise; // 观测噪声协方差矩阵

    bool hasCloudKey = false;

	ObjectStateESKF(Pose3 pose                                    = Pose3::identity(),
				Pose3 velocity                                    = Pose3::identity(),
				uint64_t poseNodeIndex                            = 0,
				uint64_t velocityNodeIndex                        = 0,
				uint64_t objectIndex                              = 0,
				uint64_t objectIndexForTracking                   = 0,
				int lostCount                                     = 0,
				int trackScore                                    = 0,
				ros::Time timestamp                               = ros::Time(),
				BoundingBox box                                   = BoundingBox(),
				BoundingBox detection                             = BoundingBox(),
				bool isTightlyCoupled                             = false,
				bool isFirst                                      = false,
				std::vector<uint64_t> previousVelocityNodeIndices = std::vector<uint64_t>())
              : pose(pose),
                velocity(velocity),
                poseNodeIndex(poseNodeIndex),
                velocityNodeIndex(velocityNodeIndex),
                objectIndex(objectIndex),
                objectIndexForTracking(objectIndexForTracking),
                lostCount(lostCount),
                trackScore(trackScore),
                timestamp(timestamp),
                box(box),
                detection(detection),
                isTightlyCoupled(isTightlyCoupled),
                isFirst(isFirst),
                previousVelocityNodeIndices(previousVelocityNodeIndices) {
	}

	ObjectStateESKF clone() const {
        ObjectStateESKF copy(pose,
                            velocity,
                            poseNodeIndex,
                            velocityNodeIndex,
                            objectIndex,
                            objectIndexForTracking,
                            lostCount,
                            trackScore,
                            timestamp,
                            box,
                            detection,
                            isTightlyCoupled,
                            isFirst,
                            previousVelocityNodeIndices);
        return copy;
    }

	bool isTurning(float threshold) const {
		auto rot = gtsam::traits<gtsam::Rot3>::Local(gtsam::Rot3::identity(), this->velocity.rotation());
		return rot.maxCoeff() > threshold;
	} // not used

	bool isMovingFast(float threshold) const {
		auto v = gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::identity(), this->velocity);
		return sqrt(pow(v(3), 2) + pow(v(4), 2) + pow(v(5), 2)) > threshold;
	} // not used

	bool velocityIsConsistent(int samplingSize,
							  Values& currentEstimates,
							  double angleThreshold,
							  double velocityThreshold) const {
		int size = previousVelocityNodeIndices.size();

		if (size < samplingSize) return false;

		Eigen::VectorXd angles     = Eigen::VectorXd::Zero(samplingSize);
		Eigen::VectorXd velocities = Eigen::VectorXd::Zero(samplingSize);
		std::vector<gtsam::Vector6> vs;
		gtsam::Vector6 vMean = gtsam::Vector6::Zero();
		for (int i = 0; i < samplingSize; ++i) {
			auto vi       = currentEstimates.at<gtsam::Pose3>(previousVelocityNodeIndices[size - i - 1]);
			auto v        = gtsam::traits<gtsam::Pose3>::Local(gtsam::Pose3::identity(), vi);
			angles(i)     = sqrt(pow(v(0), 2) + pow(v(1), 2) + pow(v(2), 2)); // 角速度模长
			velocities(i) = sqrt(pow(v(3), 2) + pow(v(4), 2) + pow(v(5), 2)); // 线速度模长
			vs.push_back(v);
			vMean += v;
		}
		vMean /= samplingSize; // 平均速度
		gtsam::Matrix6 covariance = gtsam::Matrix6::Zero();
        // 协方差矩阵
		covariance(0, 0) = covariance(1, 1) = covariance(2, 2) = angleThreshold;
		covariance(3, 3) = covariance(4, 4) = covariance(5, 5) = velocityThreshold;
		auto covarianceInverse                                 = covariance.inverse();
		double error                                           = 0.0;
		for (int i = 0; i < samplingSize; ++i) {
			auto v = vs[i] - vMean;
			error += v.transpose() * covarianceInverse * v; // 马氏距离
		}
		error /= samplingSize;

		double angleVar    = (angles.array() - angles.mean()).pow(2).mean(); // 角速度与平均角速度之间的差异的平方和的平均值
		double velocityVar = (velocities.array() - velocities.mean()).pow(2).mean(); // 线速度与平均线速度之间的差异的平方和的平均值

		// return angleVar < angleThreshold && velocityVar < velocityThreshold;

		return error < 1.0 * 1.0;
	}

    void buildNoise() {
        propagationNoise.diagonal() << 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4;
        observationNoise.diagonal() << 2.0e-4, 2.0e-4, 2.0e-4, 1.5e-3, 1.5e-3, 1.5e-3;
    }

    // 预测步骤
    void predict(double dt) {
        // Propagate the object using the constant velocity model as well as register a new variable node for the object
		auto identity = Pose3::identity();
        // template<class Class>
        // static TangentVector Local(const Class& origin, const Class& other)
        // Local 方法将两个李群 origin, other 间的相对变换映射到李代数空间，即求 ξ, 使得 origin^{-1}*other = exp(ξ)
        auto tangent_velocity = gtsam::traits<Pose3>::Local(identity, velocity);
		auto deltaPoseVec = tangent_velocity * dt; // 计算物体在 deltaTime 内的位姿变化
		// template<class Class>
        // static Class Retract(const Class& origin, const TangentVector& v) 
        // Retract 方法将李代数空间中的位姿变化量 v 映射回李群中的变换矩阵 T, 即求 T = origin ⨁ exp(v)
        auto deltaPose = gtsam::traits<Pose3>::Retract(identity, deltaPoseVec); // 表示物体在 deltaTime 内的位姿变化 T_{obj_{i-1},obj_{i}}
		pose = pose * deltaPose; // 更新物体的位姿 T_{w,obj_{i-1}}*T_{obj_{i-1},obj_{i}} --> T_{w,obj_{i}}
        
        // Mat6T F;
        // F.template block<3,3>(3,3) = Mat3T::Identity();
        // F.template block<3,3>(3,3) = gtsam::Rot3::Expmap(-tangent_velocity.tail<3>()*dt).matrix();
        // poseCovariance = F * poseCovariance * F.transpose() + propagationNoise;
    }

    // // 更新和重置步骤
    // void updateAndReset(const gtsam::Pose3& observationPose) {
    //     // update
    //     Vec6T error;
    //     error.template head<3>() = (observationPose.translation() - pose.translation()).matrix();
    //     error.template tail<3>() = gtsam::Rot3::Logmap(pose.rotation() * observationPose.rotation()).matrix();

    //     Mat6T H = Mat6T::Identity();
    //     Mat6T S = H * poseCovariance * H.transpose() + observationNoise;
    //     Mat6T K = poseCovariance * H.transpose() * S.inverse();
    //     Vec6T delta = K * error;

    //     pose = pose * gtsam::Pose3::Expmap(error);
    //     // reset
    //     Mat6T J;
    //     J.template block<3,3>(0,0) = Mat3T::Identity();
    //     J.template block<3,3>(3,3) = Mat3T::Identity() - 0.5 * gtsam::skewSymmetric(delta(3), delta(4), delta(5));
    //     poseCovariance = J * poseCovariance * J.transpose();
    // }

};

class Timer {
 private:
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> end;

 public:
	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}

	void reset() {
		start = std::chrono::high_resolution_clock::now();
	}

	void stop() {
		end = std::chrono::high_resolution_clock::now();
	}

	double elapsed() const {
		return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	}
};

class mapOptimization : public ParamServer {
 public:
	// gtsam
	NonlinearFactorGraph gtSAMgraph;
	NonlinearFactorGraph gtSAMgraphForLooselyCoupledObjects;
	Values initialEstimate;
	Values initialEstimateForLooselyCoupledObjects;
	Values optimizedEstimate;
	MaxMixtureISAM2* isam;
	Values isamCurrentEstimate;
	Eigen::MatrixXd poseCovariance;

	ros::Publisher pubLaserCloudSurround;
	ros::Publisher pubLaserOdometryGlobal;
	ros::Publisher pubLaserOdometryIncremental;
	ros::Publisher pubKeyPoses;
	ros::Publisher pubPath;
	ros::Publisher pubKeyFrameCloud;

	ros::Publisher pubHistoryKeyFrames;
	ros::Publisher pubIcpKeyFrames;
	ros::Publisher pubRecentKeyFrames;
	ros::Publisher pubRecentKeyFrame;
	ros::Publisher pubCloudRegisteredRaw;
	ros::Publisher pubLoopConstraintEdge;

	ros::Subscriber subCloud;
	ros::Subscriber subGPS;
	ros::Subscriber subLoop;
    ros::Subscriber save_path;

	ros::Publisher pubDetection;
	ros::Publisher pubLaserCloudDeskewed;
	ros::Publisher pubTrackingObjects;
	ros::Publisher pubTrackingObjectPaths;
    ros::Publisher pubTrackingObjectLabels;

	ros::Publisher pubDiagnosis;

	ros::Publisher pubReady;

	ros::ServiceServer srvSaveMap;
	ros::ServiceServer srvSaveEstimationResult;

	ros::ServiceClient detectionClient;
	lio_track::detection detectionSrv;

	std::deque<nav_msgs::Odometry> gpsQueue;
	lio_track::cloud_info cloudInfo;

	int numScan = 0;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 历史所有关键帧的角点集合(降采样后)
	vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames; // 历史所有关键帧的平面点集合(降采样后)

	pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; // 历史关键帧位移
	pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 历史关键帧位姿
	pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
	pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;
	std::vector<uint64_t> keyPoseIndices;

	pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // 当前激光帧角点集合
	pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // 当前激光帧平面点集合
	pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // 当前激光帧角点集合(降采样后)
	pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // 当前激光帧平面点集合(降采样后)
    pcl::PointCloud<PointType>::Ptr currentFrameCloud;

	pcl::PointCloud<PointType>::Ptr laserCloudOri; // 当前帧与局部地图匹配上了的角点、平面点
	pcl::PointCloud<PointType>::Ptr coeffSel; // 对应点的参数

	// 当前帧与局部地图匹配上了的角点、角点参数、角点标记
    std::vector<PointType> laserCloudOriCornerVec;
	std::vector<PointType> coeffSelCornerVec;
	std::vector<bool> laserCloudOriCornerFlag;
    // 当前帧与局部地图匹配上了的面点、面点参数、面点标记
	std::vector<PointType> laserCloudOriSurfVec;
	std::vector<PointType> coeffSelSurfVec;
	std::vector<bool> laserCloudOriSurfFlag;

	map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer; // map{，pair{角点点云, 面点点云}}
	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap; // 局部地图的角点集合
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap; // 局部地图的平面点集合
	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS; // 局部地图的角点集合(降采样后)
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS; // 局部地图的平面点集合(降采样后)

	pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap; // 局部关键帧角点点云对应的 kdtree
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap; // 局部关键帧面点点云对应的 kdtree

	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses; 
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

	pcl::VoxelGrid<PointType> downSizeFilterCorner;
	pcl::VoxelGrid<PointType> downSizeFilterSurf;
	pcl::VoxelGrid<PointType> downSizeFilterICP;
	pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;  // for surrounding key poses of scan-to-map optimization

	ros::Time timeLaserInfoStamp;
	double timeLaserInfoCur;
	double deltaTime;

	float transformTobeMapped[6];

	std::mutex mtx;
	std::mutex mtxLoopInfo;

	bool isDegenerate = false;
	cv::Mat matP;

	int laserCloudCornerFromMapDSNum = 0; // 局部地图角点数量
	int laserCloudSurfFromMapDSNum   = 0; // 局部地图平面点数量
	int laserCloudCornerLastDSNum    = 0; // 当前激光帧角点数量
	int laserCloudSurfLastDSNum      = 0; // 当前激光帧面点数量

	bool aLoopIsClosed = false; // 标志位，若存在回环因子或 GPS 因子则为 true
	map<int, int> loopIndexContainer;  // from new to old
	vector<pair<int, int>> loopIndexQueue;
	vector<gtsam::Pose3> loopPoseQueue;
	vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
	deque<std_msgs::Float64MultiArray> loopInfoVec;
    vector<ObjectStateESKF> objectAll;

	nav_msgs::Path globalPath; // 世界坐标系下的轨迹

	Eigen::Affine3f transPointAssociateToMap; // 当前帧位姿
	Eigen::Affine3f incrementalOdometryAffineFront; // 前一帧位姿
	Eigen::Affine3f incrementalOdometryAffineBack; // 当前帧位姿

	BoundingBoxArrayPtr detections;
    std::vector<Detection> detectionWithEarlyMatchingVariance;
    std::vector<Detection> detectionWithMatchingVariance;
    std::vector<Detection> detectionWithAssociationVariance;
    std::vector<Detection> detectionWithLooselyCoupledOptimizationVariance;
	std::vector<Detection> detectionWithTightlyCoupledOptimizationVariance;
	bool detectionIsActive = false;
	std::vector<std::map<uint64_t, ObjectStateESKF>> objects; // objets 是一个 vector, 其中的元素是 map, map 的键表示物体序号，值表示物体状态
	visualization_msgs::MarkerArray trackingObjectPaths;
	uint64_t numberOfRegisteredObjects = 0;
	uint64_t numberOfTrackingObjects   = 0;
	bool anyObjectIsTightlyCoupled     = false;
    visualization_msgs::MarkerArray trackingObjectLabels;

	uint64_t numberOfNodes = 0; // 计数关键帧位姿，物体位姿，物体速度等三种节点数量 

	Timer timer;
	int numberOfTightlyCoupledObjectsAtThisMoment = 0;

	mapOptimization() {
		ISAM2Params parameters;
		parameters.relinearizeThreshold = 0.1;
		parameters.relinearizeSkip      = 1;
		isam                            = new MaxMixtureISAM2(parameters);

		// 发布历史关键帧位移
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_track/mapping/trajectory", 1);
		// 发布局部关键帧地图的特征点云
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_track/mapping/map_global", 1);
		// 发布激光里程计，RViz 中表现为坐标轴
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry>("lio_track/mapping/odometry", 1);
		// 发布激光里程计，它与上面的激光里程计基本一样，只是 roll, pitch 用 IMU 数据加权平均了一下，z 做了限制
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry>("lio_track/mapping/odometry_incremental", 1);
		// 发布激光里程计路径，RViz 中表现为载体的运行轨迹 
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_track/mapping/path", 1);

        // 订阅当前激光帧点云信息，来自 featureExtraction
        subCloud = nh.subscribe<lio_track::cloud_info>("lio_track/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
		
        // 订阅 GPS 里程计
        subGPS   = nh.subscribe<nav_msgs::Odometry>(gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
		// 订阅来自外部闭环检测程序提供的闭环数据，实际未用到
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());
        save_path = nh.subscribe<nav_msgs::Odometry>("lio_track/mapping/odometry_incremental", 100, &mapOptimization::path_save, this, ros::TransportHints().tcpNoDelay());
		
        // 发布名为 lio_track/save_map 的地图保存服务 
        srvSaveMap              = nh.advertiseService("lio_track/save_map", &mapOptimization::saveMapService, this);
        // 初始化一个客户端对象 detectionClient, 调用名为 lio_track_detector 的服务
        detectionClient         = nh.serviceClient<lio_track::detection>("lio_track_detector");

		// 发布闭环匹配关键帧局部地图
        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_track/mapping/icp_loop_closure_history_cloud", 1);
		// 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_track/mapping/icp_loop_closure_corrected_cloud", 1);
		// 发布回环边，RViz 中表现为回环帧之间的连线
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_track/mapping/loop_closure_constraints", 1);

		// 发布局部地图的降采样平面点集合
        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_track/mapping/map_local", 1);
		// 发布历史帧(累加的)的角点、平面点降采样点云
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_track/mapping/cloud_registered", 1);
		// 发布当前帧原始点云配准之后的点云
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_track/mapping/cloud_registered_raw", 1);
        // 发布检测，在 RViz 中体现为绿色框
		pubDetection                  = nh.advertise<BoundingBoxArray>("lio_track/mapping/detections", 1);
		pubLaserCloudDeskewed         = nh.advertise<sensor_msgs::PointCloud2>("lio_track/mapping/cloud_deskewed", 1);

		pubTrackingObjects              = nh.advertise<BoundingBoxArray>("lio_track/tracking/objects", 1);
		pubTrackingObjectPaths          = nh.advertise<visualization_msgs::MarkerArray>("lio_track/tracking/object_paths", 1);
        pubTrackingObjectLabels         = nh.advertise<visualization_msgs::MarkerArray>("lio_track/tracking/object_labels", 1);

		pubDiagnosis = nh.advertise<lio_track::Diagnosis>("lio_track/diagnosis", 1);

		pubReady = nh.advertise<std_msgs::Empty>("lio_track/ready", 1);

		downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize); // mappingCornerLeafSize = 0.2
		downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize); // mappingSurfLeafSize = 0.2
		downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize); 
		downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity);  // for surrounding key poses of scan-to-map optimization

		allocateMemory();
	}

	void allocateMemory() {
		cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
		cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
		copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
		copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

		kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
		kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

		laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());    // corner feature set from odoOptimization
		laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());      // surf feature set from odoOptimization
		laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());  // downsampled corner featuer set from odoOptimization
		laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());    // downsampled surf featuer set from odoOptimization
        currentFrameCloud.reset(new pcl::PointCloud<PointType>());

		laserCloudOri.reset(new pcl::PointCloud<PointType>());
		coeffSel.reset(new pcl::PointCloud<PointType>());

		laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
		coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
		laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
		laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
		coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
		laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

		std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
		std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

		laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
		laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
		laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
		laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

		kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
		kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

		for (int i = 0; i < 6; ++i) {
			transformTobeMapped[i] = 0;
		}

		matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

		detections.reset(new BoundingBoxArray());
	}

	void laserCloudInfoHandler(const lio_track::cloud_infoConstPtr& msgIn) { 
        
		// 获取当前激光雷达帧时间戳
		timeLaserInfoStamp = msgIn->header.stamp; // 存储消息头中的时间戳，类型为 ros::Time
		timeLaserInfoCur   = msgIn->header.stamp.toSec(); // 将时间戳转换为秒

		// 将 ROS 消息中的角点(cloud_corner)和平面点(cloud_surface)转换为 PCL 点云格式，并分别存储在 *laserCloudCornerLast 和 *laserCloudSurfLast 中
		cloudInfo = *msgIn;
		pcl::fromROSMsg(msgIn->cloud_corner, *laserCloudCornerLast);
		pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);
        *currentFrameCloud += *laserCloudCornerLast;
        *currentFrameCloud += *laserCloudSurfLast;

		std::lock_guard<std::mutex> lock(mtx);

		timer.reset(); // 重置计时器
		numberOfTightlyCoupledObjectsAtThisMoment = 0; // 初始化当前帧进行需要紧耦合位姿优化的物体数量为 0

		static double timeLastProcessing = -1; // 控制频率稳定
		if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) { // 如果当前时间戳 timeLaserInfoCur 与上次处理时间 timeLastProcessing 的差值大于等于 mappingProcessInterval
#ifdef ENABLE_SIMULTANEOUS_LOCALIZATION_AND_TRACKING
			// std::thread::thread<_Callable, ..._Args, >(_Callable && __f, _Args && ...__args)
            // _Callable&& __f 是一个右值引用，可接收函数指针(内部应用完美转发机制），创建线程后，__f 函数会在新线程中异步执行
            // _Args&&... __args 是一个可变参数包，可以包含任意数量和类型的参数，并将这些参数传递给 __f
            std::thread t(&mapOptimization::getDetections, this); 

			deltaTime          = timeLaserInfoCur - timeLastProcessing; // 更新 deltaTime 为两帧激光雷达点云之间的时间差
			timeLastProcessing = timeLaserInfoCur; // 更新 timeLastProcessing 为当前时间戳
#endif
            // 当前帧位姿估计
			updateInitialGuess();

			// 提取局部角点、平面点云集合，加入局部地图
            extractSurroundingKeyFrames();

			// 当前激光帧角点、平面点集合降采样
            downsampleCurrentScan();

			// scan-to-map 优化当前帧位姿
            scan2MapOptimization();

#ifdef ENABLE_SIMULTANEOUS_LOCALIZATION_AND_TRACKING
		    t.join(); // 阻塞当前线程(主线程)，直到 t 线程完成其执行，以保证得到检测结果
            // std::cout << "Thread detection has finished execution." << std::endl;
#endif
			// 检查当前帧是否为关键帧并执行因子图优化
            // double t1 = ros::Time::now().toSec();
            saveKeyFramesAndFactor();
            // double t2 = ros::Time::now().toSec();
            // cout << "time of optimization is " << t2 - t1 << endl;

			// 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
            // correctPoses();

			timer.stop();

            // 发布激光里程计
            publishOdometry();

            // 发布里程计、点云、轨迹
			publishFrames();
		}
        aLoopIsClosed = false;
		pubReady.publish(std_msgs::Empty());
        numScan++;
	}

	void getDetections() {
        // double t1 = ros::Time::now().toSec();
		detectionIsActive          = false;
		detectionSrv.request.cloud = cloudInfo.cloud_raw;
		if (detectionClient.call(detectionSrv)) {
			*detections       = detectionSrv.response.detections; // 获取检测结果
			detectionIsActive = true;
		}
        // double t2 = ros::Time::now().toSec();
        // cout << "time of detection is " << t2 - t1 << endl;
	}

	void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg) {
		gpsQueue.push_back(*gpsMsg); // 向 gpsQueue 中添加数据
	}

	void pointAssociateToMap(PointType const* const pi, PointType* const po) {
		po->x         = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
		po->y         = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
		po->z         = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
		po->intensity = pi->intensity;
	} // p_{w} = R_{w,l}*p_{l} + t^{w}_{w,l}

	// 对输入点云进行坐标变换
    // cloudIn：输入的点云数据; transformIn：变换参数
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn) {

		pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>()); // 创建一个新的点云对象 cloudOut，用于存储变换后的点云数据

		int cloudSize = cloudIn->size();
		cloudOut->resize(cloudSize); // 将输出点云 cloudOut 的大小调整为与输入点云相同的大小

		Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw); // 生成变换矩阵

#pragma omp parallel for num_threads(numberOfCores) // 使用 OpenMP 并行化循环，指定使用的线程数为 numberOfCores
		for (int i = 0; i < cloudSize; ++i) {
			const auto& pointFrom         = cloudIn->points[i]; // 获取当前点 pointFrom
            // 执行变换
			cloudOut->points[i].x         = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
			cloudOut->points[i].y         = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
			cloudOut->points[i].z         = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
			cloudOut->points[i].intensity = pointFrom.intensity;
		}
		return cloudOut;
	}

	gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
		return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
							gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
	}

	gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
		return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
							gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
	}

	Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
		return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
	}

	Eigen::Affine3f trans2Affine3f(float transformIn[]) {
		return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
	}

	PointTypePose trans2PointTypePose(float transformIn[]) {
		PointTypePose thisPose6D;
		thisPose6D.x     = transformIn[3];
		thisPose6D.y     = transformIn[4];
		thisPose6D.z     = transformIn[5];
		thisPose6D.roll  = transformIn[0];
		thisPose6D.pitch = transformIn[1];
		thisPose6D.yaw   = transformIn[2];
		return thisPose6D;
	}

	// lio_track::save_mapRequest& req: 服务请求对象，包含用户指定的保存地图的目标路径和分辨率; save_mapRequest 根据 save_map.srv 自动生成，下同
    // lio_track::save_mapResponse& res: 服务响应对象，用于向调用者返回操作结果
    bool saveMapService(lio_track::save_mapRequest& req, lio_track::save_mapResponse& res) {
		string saveMapDirectory;

		cout << "****************************************************" << endl;
		cout << "Saving map to pcd files ..." << endl;
		if (req.destination.empty()) // 检查请求中的 destination 字段是否为空。如果为空，则使用默认的保存目录；否则，使用请求中提供的路径。
			saveMapDirectory = std::getenv("HOME") + savePCDDirectory; // 如果为空，则使用默认的保存目录 home/user + "/Downloads/LOAM/"
		else
			saveMapDirectory = std::getenv("HOME") + req.destination; // 将保存路径设置为用户主目录下的指定子目录 home/user + req.destination
		cout << "Save destination: " << saveMapDirectory << endl;
		// create directory and remove old files;
		int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str()); // 删除旧的保存目录及其内容
		unused     = system((std::string("mkdir -p ") + saveMapDirectory).c_str()); // 创建新的保存目录以存储新生成的文件
		// 将关键帧的位置和位姿变换保存为二进制 PCD 文件
		pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
		pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
        
		pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

		// correlationFilter(); // 关联滤除动态点

        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) { // 遍历所有关键帧，将每个关键帧的角点云和表面点云根据其位姿进行变换，并添加到全局角点点云 globalCornerCloud, 全局平面点点云 globalSurfCloud
			*globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]); // T_{w,l}*p_{l} --> p_{w}
			*globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]); // T_{w,l}*p_{l} --> p_{w}
			cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
		}

		if (req.resolution != 0) { // 如果 req 中包含降采样信息
            // 根据请求分辨率下采样 globalCornerCloud, globalSurfCloud, 成为点云 globalCornerCloudDS, globalSurfCloudDS, 并保存
			cout << "\n\nSave resolution: " << req.resolution << endl;
			// down-sample and save corner cloud
			downSizeFilterCorner.setInputCloud(globalCornerCloud);
			downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
			downSizeFilterCorner.filter(*globalCornerCloudDS);
			pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
			// down-sample and save surf cloud
			downSizeFilterSurf.setInputCloud(globalSurfCloud);
			downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
			downSizeFilterSurf.filter(*globalSurfCloudDS);
			pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
		} else { // // 如果 req 不包含降采样信息
            // 直接保存点云 globalCornerCloud, globalSurfCloud
			// save corner cloud
			pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
			// save surf cloud
			pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
		}

		// 将 globalCornerCloud, globalSurfCloud 合并为 globalMapCloud
		*globalMapCloud += *globalCornerCloud;
		*globalMapCloud += *globalSurfCloud;

		int ret     = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud); // 合并并保存全局点云地图，如果保存成功，ret 的值为 0; 如果保存失败，ret 的值为非零值，具体值取决于错误类型
		res.success = ret == 0; // 如果 ret 为 0，表示保存操作成功，res.success 被设置为 true; 如果 ret 为非零值，表示保存操作失败，res.success 被设置为 false
        // 重置下采样滤波器参数
		downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
		downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

		cout << "****************************************************" << endl;
		cout << "Saving map to pcd files completed\n" 
             << endl;

		return true;
	}

    void correlationFilter() {
        std::map<uint64_t, std::vector<ObjectStateESKF>> objectsSortByIndex;
        for (const auto& objectsOneFrame : objects) {
            for (const auto& [first, object] : objectsOneFrame) {
                objectsSortByIndex[first].push_back(object); // 按照 first 分类，将 object 存入对应的 vector
            }
        }
        for (const auto& [first, objectsForOneIndex] : objectsSortByIndex) {
            if (std::all_of(objectsForOneIndex.begin(), objectsForOneIndex.end(), [](const ObjectStateESKF& object) { return gtsam::Pose3::Logmap(object.velocity).norm() < 1.0; })) 
                continue;
            else {
                for (auto& object : objectsForOneIndex) {
                    for (int i = 0; i < (int)cloudKeyPoses6D->size(); i++) {
                        if (object.timestamp.toSec() - cloudKeyPoses6D->points[i].time < 0.01) {
                            cornerCloudKeyFrames[i]->erase(
                                std::remove_if(cornerCloudKeyFrames[i]->begin(),cornerCloudKeyFrames[i]->end(), [&](const PointType& point) { return isInBoundingBox(point, object.detection); }),
                                cornerCloudKeyFrames[i]->end());
                            surfCloudKeyFrames[i]->erase(
                                std::remove_if(surfCloudKeyFrames[i]->begin(), surfCloudKeyFrames[i]->end(), [&](const PointType& point) { return isInBoundingBox(point, object.detection); }),
                                surfCloudKeyFrames[i]->end());
                            break;
                        }
                    }
                }
            }
        }
    }

	void visualizeGlobalMapThread() { // 地图更新线程
		ros::Rate rate(0.2);
		while (ros::ok()) {
			rate.sleep();
			publishGlobalMap();
		}

		if (savePCD == false) // savePCD 默认值为 false
			return;

		lio_track::save_mapRequest req;
		lio_track::save_mapResponse res;

		if (!saveMapService(req, res)) { 
			cout << "Fail to save map" << endl;
		}
	}

	void publishGlobalMap() {
		if (pubLaserCloudSurround.getNumSubscribers() == 0)
			return;

		if (cloudKeyPoses3D->points.empty() == true)
			return;

		pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

		std::vector<int> pointSearchIndGlobalMap;
		std::vector<float> pointSearchSqDisGlobalMap;
		mtx.lock();
		kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D); // 历史关键帧位移
        // 对当前激光帧(最后一个关键帧)，在 kdtree 中搜索包含在以其为中心，半径为 globalMapVisualizationSearchRadius(1e3) 的空间范围内的关键帧，索引存入 pointSearchIndGlobalMap, 平方距离存入 pointSearchSqDisGlobalMap
		kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0); 
		mtx.unlock();
        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
			globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]); // 将与当前关键帧足够近的关键帧的位移存 globalMapKeyPoses
		
		// 降采样 globalMapKeyPoses
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;                                                                                             // for global map visualization
		downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity);  // for global map visualization
		downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
		downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
		for (auto& pt : globalMapKeyPosesDS->points) {
			kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
			pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity; // 为每个下采样后的关键帧设置强度值，强度值对应于该关键帧的索引
		}

		for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i) {
			if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
				continue;
			int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            // 将当前关键帧附近的角点和平面点云从激光雷达坐标系转换到世界坐标系，并添加到 globalMapKeyFrames
			*globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
			*globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
		}
		// 降采样 globalMapKeyFrames
		pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                    // for global map visualization
		downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize);  // for global map visualization
		downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
		downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

		publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
	}

	void loopClosureThread() { // 回环线程
		if (loopClosureEnableFlag == false) // 若 loopClosureEnableFlag 为 false, 即不进行回环检测和闭环优化
			return;

		ros::Rate rate(loopClosureFrequency); // 每 loopClosureFrequency(秒) 进行一次回环检测
		while (ros::ok()) { // ros::ok() 返回一个布尔值，用来检查当前 ROS 节点的状态
			rate.sleep();
			performLoopClosure();
			visualizeLoopClosure();
		}
	}

	void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg) {
        // not used
		std::lock_guard<std::mutex> lock(mtxLoopInfo);
		if (loopMsg->data.size() != 2)
			return;

		loopInfoVec.push_back(*loopMsg);

		while (loopInfoVec.size() > 5)
			loopInfoVec.pop_front();
	}

	void performLoopClosure() {
		if (cloudKeyPoses3D->points.empty() == true)
			return;
        
		mtx.lock();
		*copy_cloudKeyPoses3D = *cloudKeyPoses3D;
		*copy_cloudKeyPoses6D = *cloudKeyPoses6D;
		mtx.unlock();

		// find keys
		int loopKeyCur; // 当前关键帧 id
		int loopKeyPre; // 回环关键帧 id
		if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false) // 处理来自外部订阅的回环关系
			if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false) // 对当前关键帧进行回环查找，loopKeyCur 赋值为当前关键帧 id, loopKeyPre 赋值为与当前关键帧匹配的回环帧 id
				return;

		// extract cloud
		pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
		{
			loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0); // 取当前关键帧在世界坐标系下的点云，降采样后存入 cureKeyframeCloud
			loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); // 取回环帧自身及附近在世界坐标系下的点云，降采样后存入 prevKeyframeCloud
			if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
				return;
			if (pubHistoryKeyFrames.getNumSubscribers() != 0)
				publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame); // 发布回环帧附近的点云
		}

		// ICP Settings
		static pcl::IterativeClosestPoint<PointType, PointType> icp;
		icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2); // 设置最大对应点距离，超出此距离的点对将被忽略
		icp.setMaximumIterations(100);
		icp.setTransformationEpsilon(1e-6); // 两次迭代之间的变换变化小于这个值时，认为 ICP 已经收敛，停止迭代
		icp.setEuclideanFitnessEpsilon(1e-6); // 设置欧几里得误差阈值，用于评估配准的质量，且当匹配误差小于这个阈值时，认为算法已经收敛，停止迭代
		icp.setRANSACIterations(0);

		// Align clouds
		icp.setInputSource(cureKeyframeCloud); // 将降采样后的当前帧的点云作为源点云
		icp.setInputTarget(prevKeyframeCloud); // 将降采样后的回环帧自身及附近的点云作为目标点云
		pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
		icp.align(*unused_result); // 求解源点云到目标点云的最优位姿变换

		if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore) // 检查 ICP 是否收敛以及配准的质量。如果未收敛或质量低于设定阈值，则退出，认为没有检测到回环，否则认为检测到了回环
			return;

		// 发布当前关键帧经过闭环优化后的位姿变换最优位姿变换作用后的特征点云
		if (pubIcpKeyFrames.getNumSubscribers() != 0) {
			pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
			pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
			publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
		}

		// 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
		float x, y, z, roll, pitch, yaw;
		Eigen::Affine3f correctionLidarFrame;
		correctionLidarFrame = icp.getFinalTransformation(); // T_{w, cur}*T_{w, cur_wrong}^{-1}
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]); // T_{w, cur_wrong}
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // T_{w, cur}*T_{w, cur_wrong}^{-1}*T_{w, cur_wrong} --> T_{w, cur}
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z)); // T_{w, cur}
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]); // T_{w, cur_wrong}

		gtsam::Vector Vector6(6);
		float noiseScore = icp.getFitnessScore(); // 协方差使用 ICP 配准过程中得到的匹配误差
		Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
		noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

		// Add pose constraint
		mtx.lock();
        // 添加回环信息
		loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre)); // 存储闭环约束的关键帧 id 对
		loopPoseQueue.push_back(poseFrom.between(poseTo)); // T_{w, cur}^{-1}*T_{w, cur_wrong} = T_{cur, cur_wrong}
		loopNoiseQueue.push_back(constraintNoise); // 存储闭环约束的协方差
		mtx.unlock();

		// add loop constriant
		loopIndexContainer[loopKeyCur] = loopKeyPre; // 更新回环字典
	}

	bool detectLoopClosureDistance(int* latestID, int* closestID) {
		int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; // 最后一个关键帧的 id
		int loopKeyPre = -1;

		// check loop constraint added before
		auto it = loopIndexContainer.find(loopKeyCur);
		if (it != loopIndexContainer.end()) // 当前关键帧已经添加过闭环对应关系，不再继续添加
			return false;

		// 在历史关键帧中查找与当前关键帧位移之差在 historyKeyframeSearchRadius 内的关键帧索引，结果存入 pointSearchIndLoop
		std::vector<int> pointSearchIndLoop;
		std::vector<float> pointSearchSqDisLoop;
		kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
		kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0); 

		// 在候选关键帧集合中，找到与当前帧时间相隔在 historyKeyframeSearchTimeDiff 以上的最早的关键帧，设为候选匹配帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i) {
			int id = pointSearchIndLoop[i];
			if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff) { 
				loopKeyPre = id;
				break;
			}
		}

		if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
			return false;

		*latestID  = loopKeyCur;
		*closestID = loopKeyPre;

		return true;
	}

	bool detectLoopClosureExternal(int* latestID, int* closestID) {
		// this function is not used yet, please ignore it
		int loopKeyCur = -1;
		int loopKeyPre = -1;

		std::lock_guard<std::mutex> lock(mtxLoopInfo);
		if (loopInfoVec.empty())
			return false;

		double loopTimeCur = loopInfoVec.front().data[0];
		double loopTimePre = loopInfoVec.front().data[1];
		loopInfoVec.pop_front();

		if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
			return false;

		int cloudSize = copy_cloudKeyPoses6D->size();
		if (cloudSize < 2)
			return false;

		// latest key
		loopKeyCur = cloudSize - 1;
		for (int i = cloudSize - 1; i >= 0; --i) {
			if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
				loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
			else
				break;
		}

		// previous key
		loopKeyPre = 0;
		for (int i = 0; i < cloudSize; ++i) {
			if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
				loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
			else
				break;
		}

		if (loopKeyCur == loopKeyPre)
			return false;

		auto it = loopIndexContainer.find(loopKeyCur);
		if (it != loopIndexContainer.end())
			return false;

		*latestID  = loopKeyCur;
		*closestID = loopKeyPre;

		return true;
	}

	void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum) {
		// extract near keyframes
		nearKeyframes->clear();
		int cloudSize = copy_cloudKeyPoses6D->size();
         // 提取索引为 key 的关键帧前后各 searchNum 个关键帧的点云存入 nearKeyframes
		for (int i = -searchNum; i <= searchNum; ++i) {
			int keyNear = key + i;
			if (keyNear < 0 || keyNear >= cloudSize)
				continue;
			*nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); // 添加世界坐标系下角点点云到 nearKeyframes
			*nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); // 添加世界坐标系下平面点点云 nearKeyframes
		}

		if (nearKeyframes->empty())
			return;

		// 降采样 nearKeyframes
		pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
		downSizeFilterICP.setInputCloud(nearKeyframes);
		downSizeFilterICP.filter(*cloud_temp);
		*nearKeyframes = *cloud_temp;
	}

	void visualizeLoopClosure() { // 可视化回环
		if (loopIndexContainer.empty())
			return;

		visualization_msgs::MarkerArray markerArray;
		// ego loop nodes
		visualization_msgs::Marker markerNode;
		markerNode.header.frame_id    = odometryFrame;
		markerNode.header.stamp       = timeLaserInfoStamp;
		markerNode.action             = visualization_msgs::Marker::ADD;
		markerNode.type               = visualization_msgs::Marker::SPHERE_LIST;
		markerNode.ns                 = "loop_nodes";
		markerNode.id                 = 0;
		markerNode.pose.orientation.w = 1;
		markerNode.scale.x            = 0.3;
		markerNode.scale.y            = 0.3;
		markerNode.scale.z            = 0.3;
		markerNode.color.r            = 0;
		markerNode.color.g            = 0.8;
		markerNode.color.b            = 1;
		markerNode.color.a            = 1;
		// ego loop edges
		visualization_msgs::Marker markerEdge;
		markerEdge.header.frame_id    = odometryFrame;
		markerEdge.header.stamp       = timeLaserInfoStamp;
		markerEdge.action             = visualization_msgs::Marker::ADD;
		markerEdge.type               = visualization_msgs::Marker::LINE_LIST;
		markerEdge.ns                 = "loop_edges";
		markerEdge.id                 = 1;
		markerEdge.pose.orientation.w = 1;
		markerEdge.scale.x            = 0.1;
		markerEdge.color.r            = 0.9;
		markerEdge.color.g            = 0.9;
		markerEdge.color.b            = 0;
		markerEdge.color.a            = 1;
        // object loop edges
		visualization_msgs::Marker markerObjectEdge;
		markerObjectEdge.header.frame_id    = odometryFrame;
		markerObjectEdge.header.stamp       = timeLaserInfoStamp;
		markerObjectEdge.action             = visualization_msgs::Marker::ADD;
		markerObjectEdge.type               = visualization_msgs::Marker::LINE_LIST;
		markerObjectEdge.ns                 = "object_loop_edges";
		markerObjectEdge.id                 = 1;
		markerObjectEdge.pose.orientation.w = 1;
		markerObjectEdge.scale.x            = 0.1;
		markerObjectEdge.color.r            = 0.0;
		markerObjectEdge.color.g            = 1.0; // 物体回环形边为绿色
		markerObjectEdge.color.b            = 0.0;
		markerObjectEdge.color.a            = 1;

		for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it) {
			int key_cur = it->first;
			int key_pre = it->second;
			geometry_msgs::Point p;
			p.x = copy_cloudKeyPoses6D->points[key_cur].x;
			p.y = copy_cloudKeyPoses6D->points[key_cur].y;
			p.z = copy_cloudKeyPoses6D->points[key_cur].z;
			markerNode.points.push_back(p);
			markerEdge.points.push_back(p);
			p.x = copy_cloudKeyPoses6D->points[key_pre].x;
			p.y = copy_cloudKeyPoses6D->points[key_pre].y;
			p.z = copy_cloudKeyPoses6D->points[key_pre].z;
			markerNode.points.push_back(p);
			markerEdge.points.push_back(p);
		}

		markerArray.markers.push_back(markerNode);
		markerArray.markers.push_back(markerEdge);
		pubLoopConstraintEdge.publish(markerArray); // 发布回环边
	}

    void updateInitialGuess()
    {
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        if (cloudKeyPoses3D->points.empty()) // 如果 cloudKeyPoses3D->points 为空(第一个激光帧)，用当前激光帧起始时刻相对 ENU 坐标系的绝对 RPY 初始化当前帧位姿的旋转部分
        {   
            // T_{w,l_{i}}
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // 初始位姿: T_{w,l_{i}}
            std::cout << "first scan, stamp: " << cloudInfo.header.stamp << std::endl;
            return;
        }
        static bool lastImuPreTransAvailable = false; // lastImuPreTransAvailable 表示上一帧点云是否具有 IMU 预积分里程计信息，静态变量初始化只在第一次调用函数时生效
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true) // 如果当前帧点云中包含(IMU 预积分)里程计信息，首先把输入里程计记录下来
        {
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw); // T_{w, l_{i}}
            if (lastImuPreTransAvailable == false) // 当前帧点云有里程计信息但上一帧点云没有里程计信息
            {
                lastImuPreTransformation = transBack; // 因为当前帧点云含有里程计信息，故更改 lastImuPreTransformation 为当前帧位姿, 为下帧点云到来做准备
                lastImuPreTransAvailable = true; // 因为当前帧点云含有里程计信息，故更改 lastImuPreTransAvailable 为 true, 为下帧点云到来做准备
                // std::cout << "inconsectuive odom, stamp: " << cloudInfo.header.stamp << std::endl;
            } else { // 当前帧点云有里程计信息且上一帧点云也有里程计信息
                // 根据输入里程计的变化量作用到上一帧位姿上获得当前帧位姿的估计了
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack; // (T_{w,l_{i-1}}^{-1})*T_{w,l_i} --> T_{l_{i-1},l_i}
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped); // T_{w,l_{i-1}}
                Eigen::Affine3f transFinal = transTobe * transIncre; // T_{w,l_{i-1}}*T_{l_{i-1},l_i} --> T_{w,l_i}
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 将当前帧位姿存入 transformTobeMapped T_{w,l_i}

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // 将当前帧的初始变换赋值给前一帧
                // std::cout << "consectuive odom, stamp: " << cloudInfo.header.stamp << std::endl;
                return;
            }
        }
        // 当前帧有 IMU 信息
        if (cloudInfo.imuAvailable == true)
        {
            if (cloudInfo.odomAvailable == false)
                lastImuPreTransAvailable = false;
                
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // 用当前激光帧起始时刻相对 ENU 坐标系的绝对 RPY 初始化当前帧位姿的旋转部分 T(R_{earth,i_{i}},0)
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack; // T(R_{earth,i_{i-1}},0)^{-1}*T(R_{earth,i_{i}},0) --> T_{i_{i-1},i_{i}}

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped); // T_{w,i_{i-1}}
            Eigen::Affine3f transFinal = transTobe * transIncre; // T_{w,i_{i}}
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // T_{w,l_{i}} = T_{w,i_{i}}

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // T_{w, i_{i}} = T(R_{earth, i_{i}},0)
            std::cout << "assigned by imu, stamp: " << cloudInfo.header.stamp << std::endl;
            return;
        }
    }

	void extractForLoopClosure() {
		pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
		int numPoses = cloudKeyPoses3D->size();
		for (int i = numPoses - 1; i >= 0; --i) {
			if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
				cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
			else
				break;
		}

		extractCloud(cloudToExtract);
	}

	void extractNearby() {
		pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
		std::vector<int> pointSearchInd;
		std::vector<float> pointSearchSqDis;

		kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);  // cloudKeyPoses3D 表示历史所有关键帧的位移集合，通过该集合构建 kdtree
        // 对当前激光帧(最后一个关键帧)，在 kdtree 中搜索包含在以其为中心，半径为 surroundingKeyframeSearchRadius 的空间范围内的关键帧
		kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
		for (int i = 0; i < (int)pointSearchInd.size(); ++i) { // 遍历搜索结果
			int id = pointSearchInd[i];
			surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]); // 加入相邻关键帧位移集合 surroundingKeyPoses 中
		}
        // 对 surroundingKeyPoses 进行下采样，结果存入 surroundingKeyPosesDS
		downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
		downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

		for (auto& pt : surroundingKeyPosesDS->points) {
			kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
			pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity; // 获取 surroundingKeyPosesDS 中每个关键帧位移对应的反射强度值，即关键帧 id
		}

		// 在 surroundingKeyPosesDS 中加入和当前激光帧开始时间相近的关键帧的位移
		int numPoses = cloudKeyPoses3D->size();
		for (int i = numPoses - 1; i >= 0; --i) {
			if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
				surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
			else
				break;
		}

		extractCloud(surroundingKeyPosesDS); 
	}

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract) {
		// 初始化局部地图的角点集合和局部地图的面点集合
		laserCloudCornerFromMap->clear();
		laserCloudSurfFromMap->clear();
		for (int i = 0; i < (int)cloudToExtract->size(); ++i) { // 遍历和与当前激光帧时空上相近的每一个关键帧
			if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
				continue;

			int thisKeyInd = (int)cloudToExtract->points[i].intensity;
			if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) { // 如果关键帧点云已被转换到世界坐标系中
				*laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
				*laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second;
			} else { // 如果关键帧点云未被转换到世界坐标系中
				// 调用 transformPointCloud 函数，将 cornerCloudKeyFrames[thisKeyInd] 和 surfCloudKeyFrames[thisKeyInd] 转换到世界坐标系中
				pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); // T_{w,l}*p_l --> p_w
				pcl::PointCloud<PointType> laserCloudSurfTemp   = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
				*laserCloudCornerFromMap += laserCloudCornerTemp;
				*laserCloudSurfFromMap += laserCloudSurfTemp;
				laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
			}
		}

		// 对构建的角点局部地图和面点局部地图进行降采样，并保存降采样后局部地图当中点云数目
		downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
		downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
		laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
		downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
		downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
		laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

		// 若存储的关键帧数目大于 1000 则将该容器清零
		if (laserCloudMapContainer.size() > 1000)
			laserCloudMapContainer.clear();
	}

	void extractSurroundingKeyFrames() {
		if (cloudKeyPoses3D->points.empty() == true) // 若不存在关键帧，直接退出该函数
			return;

		// if (loopClosureEnableFlag == true)
		// {
		//     extractForLoopClosure();
		// } else {
		//     extractNearby();
		// }

		extractNearby();
	}

	void downsampleCurrentScan() {
		// 对降采样后的角点局部地图和面点局部地图进行降采样，并保存降采样后局部地图当中点云数目
		laserCloudCornerLastDS->clear();
		downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
		downSizeFilterCorner.filter(*laserCloudCornerLastDS);
		laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

		laserCloudSurfLastDS->clear();
		downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
		downSizeFilterSurf.filter(*laserCloudSurfLastDS);
		laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
	}

	void updatePointAssociateToMap() {
		transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
	}

	void cornerOptimization() {
		updatePointAssociateToMap(); // 更新当前位姿 T_{w,l}

#pragma omp parallel for num_threads(numberOfCores)
		for (int i = 0; i < laserCloudCornerLastDSNum; i++) { // 遍历当前帧角点集合
			PointType pointOri, pointSel, coeff;
			std::vector<int> pointSearchInd;
			std::vector<float> pointSearchSqDis;

			pointOri = laserCloudCornerLastDS->points[i]; // LiDAR 系下的角点 pointOri p_{l}
			pointAssociateToMap(&pointOri, &pointSel); // 根据当前帧位姿，将角点变换到世界坐标系下，存储在 pointSel 中 p_{w}
			kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); // 在局部角点点云中搜索距离 pointSel 最近的 5 个角点
            
			cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0)); // 协方差矩阵
			cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0)); // 特征值
			cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0)); // 特征向量

			if (pointSearchSqDis[4] < 1.0) { // 若 5 个最近邻到角点的距离都小于 1m
				// 计算 5 个点的均值坐标，记为中心点
                float cx = 0, cy = 0, cz = 0;
				for (int j = 0; j < 5; j++) {
					cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
					cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
					cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
				}
				cx /= 5;
				cy /= 5;
				cz /= 5;
                // 计算 5 个点坐标的协方差矩阵
				float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
				for (int j = 0; j < 5; j++) {
					float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
					float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
					float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

					a11 += ax * ax;
					a12 += ax * ay;
					a13 += ax * az;
					a22 += ay * ay;
					a23 += ay * az;
					a33 += az * az;
				}
				a11 /= 5;
				a12 /= 5;
				a13 /= 5;
				a22 /= 5;
				a23 /= 5;
				a33 /= 5;

				matA1.at<float>(0, 0) = a11;
				matA1.at<float>(0, 1) = a12;
				matA1.at<float>(0, 2) = a13;
				matA1.at<float>(1, 0) = a12;
				matA1.at<float>(1, 1) = a22;
				matA1.at<float>(1, 2) = a23;
				matA1.at<float>(2, 0) = a13;
				matA1.at<float>(2, 1) = a23;
				matA1.at<float>(2, 2) = a33;

				cv::eigen(matA1, matD1, matV1); // 对协方差矩阵 matA1 特征值分解，特征值存储到 matD1，特征向量存储到 matV1

				if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) { // 如果最大的特征值相比次大特征值大很多，认为这 5 个点构成了线，角点(边缘点)是合格的
					// 角点坐标 p0(即前面的 p_{w})
                    float x0 = pointSel.x;
					float y0 = pointSel.y;
					float z0 = pointSel.z;
                    // 从中心点沿着最大的特征值对应的特征向量方向(直线方向)，前后各取一个点
                    // p1
					float x1 = cx + 0.1 * matV1.at<float>(0, 0);
					float y1 = cy + 0.1 * matV1.at<float>(0, 1);
					float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    // p2
					float x2 = cx - 0.1 * matV1.at<float>(0, 0);
					float y2 = cy - 0.1 * matV1.at<float>(0, 1);
					float z2 = cz - 0.1 * matV1.at<float>(0, 2);

					// p0, p1, p2 三个点组成的三角形面积的两倍; ║v║, v = (p0 - p1)×(p0 - p2)
					float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + // v_z
									  ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +  // v_y
									  ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))); // v_x

					// p1, p2 两个点的距离，也即三角形的底边长度; ║w║, w = p1 - p2
                    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

					// 三角形的高的两倍，也即 p0 到 p1 和 p2 组成的底边的直线距离的两倍; e = ║v║/║w║
                    float ld2 = a012 / l12;

                    // ∂e/∂(x0)
                    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + 
                                (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

					// ∂e/∂(y0)
                    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - 
                                 (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

					// ∂e/∂(z0)
                    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + 
                                 (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2); // s 表示权重，点到直线的距离越近，权重的值越大

					// 加权偏导数
                    coeff.x         = s * la;
					coeff.y         = s * lb;
					coeff.z         = s * lc;
                    // 加权的点到直线距离
					coeff.intensity = s * ld2;

					if (s > 0.1) { // 权重的值大于 0.1，即角点到直线的距离足够近
                        // 添加当前帧与局部地图匹配上了的角点、角点参数、角点标记
						laserCloudOriCornerVec[i]  = pointOri;
						coeffSelCornerVec[i]       = coeff;
						laserCloudOriCornerFlag[i] = true;
					}
				}
			}
		}
	}

	void surfOptimization() {
		updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
		for (int i = 0; i < laserCloudSurfLastDSNum; i++) {
			PointType pointOri, pointSel, coeff;
			std::vector<int> pointSearchInd;
			std::vector<float> pointSearchSqDis;

			pointOri = laserCloudSurfLastDS->points[i];
			pointAssociateToMap(&pointOri, &pointSel);
			kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); // 在局部面点点云中搜索距离 pointSel 最近的 5 个角点

			Eigen::Matrix<float, 5, 3> matA0;
			Eigen::Matrix<float, 5, 1> matB0;
			Eigen::Vector3f matX0;

			matA0.setZero();
			matB0.fill(-1);
			matX0.setZero();

			if (pointSearchSqDis[4] < 1.0) {
				for (int j = 0; j < 5; j++) {
					matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
					matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
					matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
				}
                // 解方程 matA0*matX0 = matB0; 通过多组 (x,y,z) 值求解平面方程 Ax + By + Cz + 1 = 0 的系数 A, B, C
				matX0 = matA0.colPivHouseholderQr().solve(matB0);

				// 平面方程的系数，也是法向量的分量
                float pa = matX0(0, 0);
				float pb = matX0(1, 0);
				float pc = matX0(2, 0);
				float pd = 1;

				float ps = sqrt(pa * pa + pb * pb + pc * pc); // sqrt(A^2 + B^2 + C^2)
				pa /= ps; // A/sqrt(A^2 + B^2 + C^2)
				pb /= ps; // B/sqrt(A^2 + B^2 + C^2)
				pc /= ps; // C/sqrt(A^2 + B^2 + C^2)
				pd /= ps; // 1/sqrt(A^2 + B^2 + C^2)

				// 检查平面是否有效，如果 5 个点中有点到平面的距离超过 0.2m, 那么认为这些点太分散了，平面无效
                bool planeValid = true;
				for (int j = 0; j < 5; j++) {
					if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
							 pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
							 pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
						planeValid = false;
						break;
					}
				}

				if (planeValid) { // 如果平面有效
					float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd; // 面点到平面距离 e = pa*x0 + pb*y0 + pc*z0 + pd

					float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z)); // 权重，点到平面的距离越近，权重的值越大

					coeff.x         = s * pa; // pa = ∂e/∂(x0)
					coeff.y         = s * pb; // pb = ∂e/∂(y0)
					coeff.z         = s * pc; // pb = ∂e/∂(z0)
					coeff.intensity = s * pd2;

					if (s > 0.1) {
                        // 更新与当前帧与局部地图匹配上了的面点、面点参数、面点标记
						laserCloudOriSurfVec[i]  = pointOri;
						coeffSelSurfVec[i]       = coeff;
						laserCloudOriSurfFlag[i] = true;
					}
				}
			}
		}
	}

	void combineOptimizationCoeffs() {
        // 提取角点，面点和二者对应参数到统一容器
		for (int i = 0; i < laserCloudCornerLastDSNum; ++i) { // 遍历当前帧角点集合，提取出与局部地图匹配上了的角点和角点参数到 laserCloudOri 和 coeffSel
			if (laserCloudOriCornerFlag[i] == true) {
				laserCloudOri->push_back(laserCloudOriCornerVec[i]);
				coeffSel->push_back(coeffSelCornerVec[i]);
			}
		}
		for (int i = 0; i < laserCloudSurfLastDSNum; ++i) { // 遍历当前帧面点集合，提取出与局部地图匹配上了的面点和面点参数到 laserCloudOri 和 coeffSel
			if (laserCloudOriSurfFlag[i] == true) {
				laserCloudOri->push_back(laserCloudOriSurfVec[i]);
				coeffSel->push_back(coeffSelSurfVec[i]);
			}
		}
		// 清空标记
		std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
		std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
	}

	bool LMOptimization(int iterCount) {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

		float srx = sin(transformTobeMapped[1]); // sin(α_new)
		float crx = cos(transformTobeMapped[1]); // cos(α_new)
		float sry = sin(transformTobeMapped[2]); // sin(β_new)
		float cry = cos(transformTobeMapped[2]); // cos(β_new)
		float srz = sin(transformTobeMapped[0]); // sin(γ_new)
		float crz = cos(transformTobeMapped[0]); // cos(γ_new)

		int laserCloudSelNum = laserCloudOri->size();
		if (laserCloudSelNum < 50) { // 当前帧匹配特征点数太少，退出
			return false;
		}

		cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0)); // 雅可比矩阵 J
		cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
		cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
		cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0)); // 残差矩阵 e
		cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
		cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0)); // 待求解 6 自由度位姿增量 ΔT

		PointType pointOri, coeff;

		// 遍历匹配特征点，构建雅可比矩阵
        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
			pointOri.x = laserCloudOri->points[i].y;
			pointOri.y = laserCloudOri->points[i].z;
			pointOri.z = laserCloudOri->points[i].x;
			// lidar -> camera
			coeff.x         = coeffSel->points[i].y;
			coeff.y         = coeffSel->points[i].z;
			coeff.z         = coeffSel->points[i].x;
			coeff.intensity = coeffSel->points[i].intensity;
			// in camera
            // Ry(β_new)Rx(α_new)Rz(γ_new)p_{c} + t^{w}_{w,c} --> p_{w} --> e; 记 t^{w}_{w,c} = (x_new, y_new, z_new)
            // ∂e/∂(α_new)
			float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;
            // ∂e/∂(β_new)
			float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;
            // ∂e/∂(γ_new)
			float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;
			// camera -> lidar
			matA.at<float>(i, 0) = arz; // ∂e/∂α
			matA.at<float>(i, 1) = arx; // ∂e/∂β
			matA.at<float>(i, 2) = ary; // ∂e/∂γ
			matA.at<float>(i, 3) = coeff.z; // ∂e/∂x
			matA.at<float>(i, 4) = coeff.x; // ∂e/∂y
			matA.at<float>(i, 5) = coeff.y; // ∂e/∂z
			matB.at<float>(i, 0) = -coeff.intensity; // 取误差(点到直线、平面距离)的负值
		}

		cv::transpose(matA, matAt);
		matAtA = matAt * matA;
		matAtB = matAt * matB;
		cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR); // 解方程 matAtA*matX + matAtB = 0; (J^{T}J)*ΔT = -J^{T}e

		if (iterCount == 0) { // 首次迭代，检查近似 Hessian 矩阵 J^{T}J 是否退化
			cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
			cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
			cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

			cv::eigen(matAtA, matE, matV); // 计算矩阵 matAtA 的特征值和特征向量，分别存储在 matE 和 matV 中
			matV.copyTo(matV2); // 将特征向量矩阵 matV 复制到 matV2

			isDegenerate      = false; // 标记矩阵是否退化
			float eignThre[6] = {100, 100, 100, 100, 100, 100}; // 特征值的阈值数组
            // 循环从最后一个特征值开始向前检查，如果某个特征值小于对应的阈值，则将对应的特征向量在 matV2 中置零，并标记 isDegenerate 为 true
			for (int i = 5; i >= 0; i--) {
				if (matE.at<float>(0, i) < eignThre[i]) {
					for (int j = 0; j < 6; j++) {
						matV2.at<float>(i, j) = 0;
					}
					isDegenerate = true;
				} else { // 遇到特征值大于或等于阈值，循环终止
					break;
				}
			}
			matP = matV.inv() * matV2;
		}

		if (isDegenerate) {
			cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
			matX.copyTo(matX2);
            // 使用投影矩阵 matP 修正位姿增量 matX
            // 投影矩阵 matP 的作用是将向量 matX2 投影到非退化方向的子空间上。具体来说，matP 将 matX2 在退化方向上的分量置零，只保留非退化方向上的分量
			matX = matP * matX2;
		}
        
        // 更新位姿
		transformTobeMapped[0] += matX.at<float>(0, 0);
		transformTobeMapped[1] += matX.at<float>(1, 0);
		transformTobeMapped[2] += matX.at<float>(2, 0);
		transformTobeMapped[3] += matX.at<float>(3, 0);
		transformTobeMapped[4] += matX.at<float>(4, 0);
		transformTobeMapped[5] += matX.at<float>(5, 0);

		float deltaR = sqrt(
				pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
				pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
				pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
		float deltaT = sqrt(
				pow(matX.at<float>(3, 0) * 100, 2) +
				pow(matX.at<float>(4, 0) * 100, 2) +
				pow(matX.at<float>(5, 0) * 100, 2));

		if ((deltaR < 0.05) && (deltaT < 0.05)) { // 若 delta_R 或 deltaT 很小，认为收敛
			return true;  // converged
		}
		return false;  // keep optimizing
	}

    void scan2MapOptimization() {
		if (cloudKeyPoses3D->points.empty())
			return;

		if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum) { // 如果当前激光帧的角点、平面点数量足够多
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS); // 局部角点点云构建 kdtree
			kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS); // 局部面点点云构建 kdtree

			for (int iterCount = 0; iterCount < 30; iterCount++) {
				// 每次迭代清空特征点集合
                laserCloudOri->clear();
				coeffSel->clear();

				cornerOptimization(); // 当前激光帧与局部地图之间的角点-角点特征匹配
				surfOptimization(); // 当前激光帧与局部地图之间的面点-面点特征匹配

				combineOptimizationCoeffs(); // 提取当前帧中与局部地图匹配上了的角点、平面点，加入同一集合

				if (LMOptimization(iterCount) == true) // scan-to-map 优化
					break;
			}

			transformUpdate(); // 更新当前帧位姿
		} else {
			ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
		}
	}

	void transformUpdate() {
		if (cloudInfo.imuAvailable == true) {
			if (std::abs(cloudInfo.imuPitchInit) < 1.4) { // 俯仰角弧度小于 1.4
				double imuWeight = imuRPYWeight; // IMU 数据的权重，用于控制IMU数据在融合中的影响 0.01
				tf::Quaternion imuQuaternion; // IMU 数据
				tf::Quaternion transformQuaternion; // 当前变换的四元数表示
				double rollMid, pitchMid, yawMid; // 用于存储中间计算结果的欧拉角

				// roll 角求加权均值，用 scan-to-map 优化得到的位姿与 IMU 原始 RPY 数据，进行加权平均
				transformQuaternion.setRPY(transformTobeMapped[0], 0, 0); // 将当前变换的 roll 角转换为四元数 transformQuaternion
				imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0); // 将IMU的 roll 角转换为四元数 imuQuaternion
				tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid); // 使用球面线性插值(Slerp)将两个四元数进行插值，并将插值后的四元数转换回欧拉角
				transformTobeMapped[0] = rollMid; // 更新 roll 角

				// pitch 角求加权均值，用 scan-to-map 优化得到的位姿与 IMU 原始 RPY 数据，进行加权平均
				transformQuaternion.setRPY(0, transformTobeMapped[1], 0); // 将当前变换的 pitch 角转换为四元数 transformQuaternion
				imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0); // 将IMU的 pitch 角转换为四元数 imuQuaternion
				tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid); // 使用球面线性插值(Slerp)将两个四元数进行插值，并将插值后的四元数转换回欧拉角
				transformTobeMapped[1] = pitchMid; // 更新 pitch 角
			}
		}

        // 对 roll 角、pitch 角和 Z 轴平移进行约束，确保它们在合理的范围内
		transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance); // FLT_MAX，即无约束
		transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance); // FLT_MAX
		transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance); // FLT_MAX

		incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped); // 当前帧位姿
	}

	float constraintTransformation(float value, float limit) {
		if (value < -limit)
			value = -limit;
		if (value > limit)
			value = limit;

		return value;
	}

	bool saveFrame() { 
		if (cloudKeyPoses3D->points.empty())
			return true;

		Eigen::Affine3f transStart   = pclPointToAffine3f(cloudKeyPoses6D->back()); // 前一帧位姿
		Eigen::Affine3f transFinal   = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
															  transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 当前帧位姿
		Eigen::Affine3f transBetween = transStart.inverse() * transFinal; // 前一帧和当前帧的相对位姿变换
		float x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); // 提取相对变换的平移和欧拉角
 
		// 旋转和平移量都较小，当前帧不设为关键帧
        if (abs(roll) < surroundingkeyframeAddingAngleThreshold && // 1.0
			abs(pitch) < surroundingkeyframeAddingAngleThreshold && // 0.2
			abs(yaw) < surroundingkeyframeAddingAngleThreshold && // 1.0
			sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold) // 50.0
			return false;

		return true; // 否则当前帧设为关键帧
	}

	void addOdomFactor() {
		if (cloudKeyPoses3D->points.empty()) { // 第一帧
			auto currentKeyIndex = numberOfNodes++;  // 该行代码会改变 numberOfNodes 的值 0++
			keyPoseIndices.push_back(currentKeyIndex);

			noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances(priorOdometryDiagonalVarianceEigenVector);  // 先验因子的协方差矩阵，此处设的很大从而固定初始值 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8
			gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise)); // 向因子图中添加先验因子
			initialEstimate.insert(currentKeyIndex, trans2gtsamPose(transformTobeMapped));
		} else { // 非首帧不添加先验因子。这是因为回环因子和先验因子协方差矩阵相差较大，添加先验因子会导致回环作用较弱
			auto previousKeyIndex = keyPoseIndices.back();
			auto currentKeyIndex  = numberOfNodes++;
			keyPoseIndices.push_back(currentKeyIndex);

			noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances(odometryDiagonalVarianceEigenVector); // 激光里程计因子的协方差矩阵 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4
			gtsam::Pose3 poseFrom                          = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); // 前一关键帧位姿
			gtsam::Pose3 poseTo                            = trans2gtsamPose(transformTobeMapped); // 当前帧位姿
			gtSAMgraph.add(BetweenFactor<Pose3>(previousKeyIndex, // 前一关键帧 id, 该 id 对应优化器内部的一个位姿变量 T_{i-1}
												currentKeyIndex, // 当前关键帧 id, 该 id 对应优化器内部的一个位姿变量 T_{i}
												poseFrom.between(poseTo), // 在优化过程中，优化器会尽量使 (T_{i-1}^{-1})*T_{i} 的估计值接近 poseFrom^{-1}*poseTo, 但也会考虑其他因子的影响
												odometryNoise));
			initialEstimate.insert(currentKeyIndex, poseTo); // 当前帧位姿初始值
		}
	}

	void addGPSFactor() {
		if (gpsQueue.empty())
			return;

		if (cloudKeyPoses3D->points.empty()) // 如果没有关键帧，不添加 GPS 因子
			return;
		else {
			if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0) // 如果第一个关键帧关键与最后一个关键帧距离小于 5m, 说明刚起步或者可能触发回环，不添加 GPS 因子
				return;
		}

		if (poseCovariance(3, 3) < poseCovThreshold && poseCovariance(4, 4) < poseCovThreshold) // 如果位姿协方差很小，说明位姿已足够准确，不添加 GPS 因子
			return;

		// last gps position
		static PointType lastGPSPoint;

		while (!gpsQueue.empty()) {
			if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2) { // 删除当前帧时间 0.2s 之前的里程计
				// message too old
				gpsQueue.pop_front();
			} else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2) { // 超过当前帧时间 0.2s 之后，退出循环
				// message too new
				break;
			} else {
				nav_msgs::Odometry thisGPS = gpsQueue.front();
				gpsQueue.pop_front();

				// GPS too noisy, skip
				float noise_x = thisGPS.pose.covariance[0];
				float noise_y = thisGPS.pose.covariance[7];
				float noise_z = thisGPS.pose.covariance[14];
				if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)  // GPS 数据协方差太大，不添加 GPS 因子
					continue;

				// GPS里程计位置
                float gps_x = thisGPS.pose.pose.position.x;
				float gps_y = thisGPS.pose.pose.position.y;
				float gps_z = thisGPS.pose.pose.position.z;
				if (!useGpsElevation) { // 若不使用GPS的z信息， 则使用雷达里程计的初始估计值
					gps_z   = transformTobeMapped[5];
					noise_z = 0.01;
				}

				// GPS not properly initialized (0,0,0)
				if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6) // 如果 GPS 的 x,y 值过小则说明 GPS 还未初始化，不添加 GPS 因子
					continue;

				// Add GPS every a few meters
				PointType curGPSPoint;
				curGPSPoint.x = gps_x;
				curGPSPoint.y = gps_y;
				curGPSPoint.z = gps_z;
				if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0) // 若两个 GPS 数据之间间隔小于 5m, 为避免过于频繁地添加 GPS 因子，此处不添加 GPS 因子
					continue;
				else
					lastGPSPoint = curGPSPoint;

				gtsam::Vector Vector3(3);
				Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
				noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
				gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise); // 添加GPS因子
				gtSAMgraph.add(gps_factor);

				aLoopIsClosed = true;
				break;
			}
		}
	}

    void addLoopFactor() {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i) {
            int indexFrom                                        = loopIndexQueue[i].first;
            int indexTo                                          = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween                             = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void propagateObjectPoses() {
		std::map<uint64_t, ObjectStateESKF> nextObjects; // 存储下一时间步的物体状态

		// 检查 objects 是否为空。如果是，则直接将空的 nextObjects 添加到 objects 列表中，并返回
		if (objects.empty()) {
			objects.push_back(nextObjects);
			return;
		}

#ifdef MAP_OPTIMIZATION_DEBUG
		std::cout << "DELTA_TIME :: " << deltaTime << "\n\n";
#endif
		for (const auto& pairedObject : objects.back()) { // 遍历上一时间步的物体
			if (pairedObject.second.lostCount > trackingStepsForLostObject) continue; // 如果物体丢失的次数超过了 trackingStepsForLostObject, 则跳过该物体

			// Propagate the object using the constant velocity model as well as register a new variable node for the object
			auto identity     = Pose3::identity();
			auto nextObject   = pairedObject.second.clone(); // nextObject 赋值为上一时间步物体的状态 T_{w,obj_{i-1}}
            
            nextObject.predict(deltaTime);

			nextObject.isFirst   = false;
			nextObject.timestamp = timeLaserInfoStamp;
			
            if (pairedObject.second.lostCount == 0) { // 如果物体未丢失
				// 分别为物体的位姿和速度分配新的节点索引
                nextObject.poseNodeIndex     = numberOfNodes++;
				nextObject.velocityNodeIndex = numberOfNodes++;
				// 将物体的位姿和速度插入到初始估计值中
                initialEstimate.insert(nextObject.poseNodeIndex, nextObject.pose);
				initialEstimate.insert(nextObject.velocityNodeIndex, nextObject.velocity);

				nextObject.previousVelocityNodeIndices.push_back(pairedObject.second.velocityNodeIndex); // 保存上一个时间步的速度节点索引，用于后续的约束添加
			} else { // 如果物体丢失，则将 poseNodeIndex 和 velocityNodeIndex 设置为 -1, 表示物体不参与图优化
				nextObject.poseNodeIndex     = -1;
				nextObject.velocityNodeIndex = -1;
			}

            // if (pairedObject.second.objectCloud && !pairedObject.second.objectCloud->empty()) {
            //     // 把上一帧点云变换到预测位姿
            //     Eigen::Matrix4f tf = (nextObject.pose.matrix().cast<float>());
            //     pcl::PointCloud<PointType>::Ptr propagated(new pcl::PointCloud<PointType>());
            //     pcl::transformPointCloud(*pairedObject.second.objectCloud, *propagated, tf);
            //     nextObject.objectCloud = propagated;
            // }

			nextObjects[pairedObject.first] = nextObject; // 将预测后的物体状态添加到 nextObjects 中
		}
		objects.push_back(nextObjects); // 将 nextObjects 添加到 objects 中作为当前时间步的物体初始估计(objects 中不包含上一时间步中 lostCount 大于 3 的物体)

#ifdef ENABLE_MINIMAL_MEMORY_USAGE // 如果启用内存优化 FALSE
		if (objects.size() > 2 &&
			objects.size() > numberOfPreLooseCouplingSteps + 1 &&
			objects.size() > numberOfVelocityConsistencySteps + 1) { // objects 列表的大小超过了某些阈值，则删除最早的物体状态
		        objects.erase(objects.begin()); // 删除最早的物体状态
		}
#endif
	}

	void addConstantVelocityFactor() {
		if (objects.size() < 2) return; // 如果 objects 列表中少于两个时间步的状态，则直接返回，不进行后续处理 // Skip the process if there is no enough time stamps for adding constant

		for (const auto& pairedObject : objects.back()) { // 遍历当前时间步的物体
			if (pairedObject.second.isFirst) continue; // 如果物体是第一次出现，跳过该物体
			if (pairedObject.second.lostCount > 0) continue; // 如果物体在当前时间步已经丢失，跳过该物体

			auto noiseModel      = noiseModel::Diagonal::Variances(constantVelocityDiagonalVarianceEigenVector); // 速度噪声模型(协方差），该噪声模型用于大多数情况下
			auto earlyNoiseModel = noiseModel::Diagonal::Variances(earlyConstantVelocityDiagonalVarianceEigenVector); // 早期速度噪声模型，该噪声模型用于物体路径的前几个时间步，通常具有更大的方差，以允许更多的不确定性
			auto currentObject   = pairedObject.second; // 物体状态
			auto objectIndex     = currentObject.objectIndex; // 物体索引
			auto previousObject  = objects[objects.size() - 2][objectIndex]; // 上一时间步同一物体的状态
			if (pairedObject.second.isTightlyCoupled) { // 若物体和自车之间是紧耦合的，则将恒定速度因子添加到主因子图 gtSAMgraph 中，使用 noiseModel 作为噪声模型
				gtSAMgraphForLooselyCoupledObjects.add(ConstantVelocityFactor(previousObject.velocityNodeIndex,
													  currentObject.velocityNodeIndex,
													  noiseModel));
			} else { // 若物体状态之间是松耦合的，则根据物体路径的长度来决定使用哪种噪声模型
				if (trackingObjectPaths.markers[pairedObject.second.objectIndex].points.size() <= numberOfEarlySteps) { // 如果物体路径的点数小于等于 numberOfEarlySteps, 则使用 earlyNoiseModel 作为噪声模型，将其添加到 gtSAMgraphForLooselyCoupledObjects 中
					gtSAMgraphForLooselyCoupledObjects.add(ConstantVelocityFactor(previousObject.velocityNodeIndex,
																				  currentObject.velocityNodeIndex,
																				  earlyNoiseModel));
				} else { // 否则，使用 noiseModel 作为噪声模型，仍将其添加到 gtSAMgraphForLooselyCoupledObjects 中
					gtSAMgraphForLooselyCoupledObjects.add(ConstantVelocityFactor(previousObject.velocityNodeIndex,
																				  currentObject.velocityNodeIndex,
																				  noiseModel));
				}
			}
		}
	}

	void addStablePoseFactor() {
		if (objects.size() < 2) return;

		for (auto& pairedObject : objects.back()) { // 遍历当前时间步的物体
			if (pairedObject.second.isFirst) continue;
			if (pairedObject.second.lostCount > 0) continue;

			auto noise          = noiseModel::Diagonal::Variances(motionDiagonalVarianceEigenVector); // 运动噪声模型
			auto& currentObject = pairedObject.second; // 物体状态
			auto objectIndex    = currentObject.objectIndex; // 物体索引
			auto previousObject = objects[objects.size() - 2][objectIndex]; // 上一时间步同一物体的状态
			if (pairedObject.second.isTightlyCoupled) { // 若物体和自车位姿之间是紧耦合的，则将平滑运动因子添加到主因子图 gtSAMgraph 中
				gtSAMgraphForLooselyCoupledObjects.add(StablePoseFactor(previousObject.poseNodeIndex, // 上一时间步的位姿节点索引
#ifdef ENABLE_COMPACT_VERSION_OF_FACTOR_GRAPH
										        currentObject.velocityNodeIndex, // 当前或上一时间步的速度节点索引。具体选择取决于是否启用了紧凑版因子图(通过 ENABLE_COMPACT_VERSION_OF_FACTOR_GRAPH 宏控制)
#else
										        previousObject.velocityNodeIndex,
#endif
										        currentObject.poseNodeIndex, // 当前时间步的位姿节点索引
										        deltaTime, // 时间间隔
										        noise)); // 运动噪声模型
			} else { // 若物体状态之间是松耦合的，则将平滑运动因子添加到因子图 gtSAMgraphForLooselyCoupledObjects 中
				gtSAMgraphForLooselyCoupledObjects.add(StablePoseFactor(previousObject.poseNodeIndex,
#ifdef ENABLE_COMPACT_VERSION_OF_FACTOR_GRAPH
			                                                            currentObject.velocityNodeIndex,
#else
				                                                        previousObject.velocityNodeIndex,
#endif
				                                                        currentObject.poseNodeIndex,
				                                                        deltaTime,
				                                                        noise));
			}
			currentObject.motionFactorPtr = boost::make_shared<StablePoseFactor>(previousObject.poseNodeIndex, 
#ifdef ENABLE_COMPACT_VERSION_OF_FACTOR_GRAPH
				                                                                 currentObject.velocityNodeIndex,
#else
				                                                                 previousObject.velocityNodeIndex,
#endif
				                                                                 currentObject.poseNodeIndex,
				                                                                 deltaTime,
				                                                                 noise); // 创建一个 StablePoseFactor 的共享指针，并将其保存在 currentObject 中
		}
	}

	void addDetectionFactor(bool requiredMockDetection = false) {
		anyObjectIsTightlyCoupled = false;

		if (detections->boxes.size() == 0 && objects.size() == 0) { // 如果当前时间步没有检测到任何物体，并且 objects 列表为空
			return;
		} else if (detections->boxes.size() == 0 && objects.size() > 0) { // 如果当前时间步没有检测到任何物体，但 objects 列表非空
			for (auto& pairedObject : objects.back()) { // 遍历当前时间步的物体
				auto& object = pairedObject.second; 
				++object.lostCount; // 将物体标记为丢失
			}
			return;
		}
        // 下面处理当前时间步至少检测到一个物体的情况
		Eigen::MatrixXi indicator = Eigen::MatrixXi::Zero(objects.back().size() + 1, detections->boxes.size()); // 当前时间步的物体和检测的关联矩阵 (N+1)×M; N+1 是为了避免 N=0 导致的无定义
		std::vector<int> trackingObjectIndices(detections->boxes.size(), -1); // trackingObjectIndices 记录每个检测对应的物体索引 M×1, 其中全体值初始化为 -1
        std::vector<int> matchingError(detections->boxes.size(), std::numeric_limits<int>::max());
        std::vector<ObjectStateESKF *> matchingObjectPtr(detections->boxes.size(), nullptr);
        std::vector<gtsam::Pose3> trackingObjectVelocity(detections->boxes.size(), Pose3::identity());

		auto smallEgoMotion = Pose3(Rot3::RzRyRx(0, 0, 0), Point3(0, 0, 0)); // 自车位姿变换
		if (requiredMockDetection) { // 如果需要模拟检测
			Eigen::Affine3f transStart   = pclPointToAffine3f(cloudKeyPoses6D->back()); // 上一关键帧位姿 T_{w,l_{k}}
			Eigen::Affine3f transFinal   = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
																  transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // T_{w,l_{i}}
			Eigen::Affine3f transBetween = transStart.inverse() * transFinal; // T_{l_{k},l_{i}}
			float x, y, z, roll, pitch, yaw;
			pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);
			smallEgoMotion = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z)); // T_{l_{k},l_{i}}
		}
        
        detectionWithEarlyMatchingVariance.clear();
        detectionWithMatchingVariance.clear();
        detectionWithAssociationVariance.clear();
        detectionWithLooselyCoupledOptimizationVariance.clear();
		detectionWithTightlyCoupledOptimizationVariance.clear();

		for (auto box : detections->boxes) { // 循环遍历每个检测框
			if (requiredMockDetection) { // 如果需要模拟检测
				auto pose = Pose3(Rot3::Quaternion(box.pose.orientation.w, box.pose.orientation.x, box.pose.orientation.y, box.pose.orientation.z),
								  Point3(box.pose.position.x, box.pose.position.y, box.pose.position.z)); 
				pose      = smallEgoMotion * pose; // T_{l_{k},l_{i}}*T_{l_{i},obj_{i,j}} --> T_{l_{k},obj_{i,j}} // to do: 此处将物体位姿变换作用到检测结果上，引入了额外误差

				auto quat              = pose.rotation().toQuaternion(); 
				box.pose.orientation.w = quat.w();
				box.pose.orientation.x = quat.x();
				box.pose.orientation.y = quat.y();
				box.pose.orientation.z = quat.z();

				auto pos            = pose.translation();
				box.pose.position.x = pos.x();
				box.pose.position.y = pos.y();
				box.pose.position.z = pos.z();
			}
			detectionWithEarlyMatchingVariance.emplace_back(box, earlyLooselyCoupledMatchingVarianceEigenVector); // [3.0e-4, 3.0e-4, 3.0e-4, 5.0e-2, 5.0e-3, 5.0e-3]
            detectionWithMatchingVariance.emplace_back(box, looselyCoupledMatchingVarianceEigenVector); // [1.0e-4, 1.0e-4, 1.0e-4, 2.0e-3, 2.0e-3, 2.0e-3]
            detectionWithAssociationVariance.emplace_back(box, dataAssociationVarianceEigenVector); // [3.0e-4, 3.0e-4, 3.0e-4, 5.0e-2, 3.0e-2, 3.0e-2]
            detectionWithLooselyCoupledOptimizationVariance.emplace_back(box, looselyCoupledDetectionVarianceEigenVector); // [2.0e-4, 2.0e-4, 2.0e-4, 1.5e-3, 1.5e-3, 1.5e-3]
			detectionWithTightlyCoupledOptimizationVariance.emplace_back(box, tightlyCoupledDetectionVarianceEigenVector); // [2.0e-4, 2.0e-4, 2.0e-4, 1.5e-3, 1.5e-3, 1.5e-3]
		} // 将全部的检测结果存入相应的 vector 中，后续将根据这些 vector 匹配物体与检测结果

		auto egoPoseKey = keyPoseIndices.back(); // 最后一个关键帧的 id = k
		auto egoPose    = initialEstimate.at<Pose3>(egoPoseKey); // 当前帧自车位姿初始值 T_{w,l_{k}} 
		auto invEgoPose = egoPose.inverse(); // T_{w,l_{k}}^{-1}

#ifdef MAP_OPTIMIZATION_DEBUG // 宏定义，用于控制是否打印调试信息。如果定义了该宏，则会输出当前物体的索引
		std::cout << "OBJECT_INDICES ::\n";
#endif
		size_t i = 0;  // 物体索引 0,...,N-1
		for (auto& pairedObject : objects.back()) { // 遍历当前时间步的每个物体
			auto& object = pairedObject.second;
			size_t j;  // 最优检测索引 0,...,M-1
			double error;

            size_t dataAssociationJ;
            double dataAssociationError;

            bool changeMatchingObject = false;

#ifdef MAP_OPTIMIZATION_DEBUG
			std::cout << object.objectIndex << ' ';
#endif
			auto&& predictedPose = invEgoPose * object.pose; // 物体检测位姿的预测 T_{w,l_{k}}^{-1}*T_{w,obj_{i,j}} --> T_{l_{k},obj_{i,j}}
            double volume = object.detection.dimensions.x * object.detection.dimensions.y * object.detection.dimensions.z;
            auto identity = Pose3::identity();
            double linearVelocityScale = gtsam::traits<Pose3>::Local(identity, object.velocity).head<3>().norm(); // gtsam::Pose3::Logmap(object.velocity).tail<3>().norm();
            bool isFast = linearVelocityScale > 0.5;
            // if(isFast) {
            //     cout << "the linear velocity scale of " << object.objectIndexForTracking << "th object is " << linearVelocityScale << "and it is fast" << endl;
            // }
            // else {
            //     cout << "the linear velocity scale of " << object.objectIndexForTracking << "th object is " << linearVelocityScale << "and it is slow." << endl;
            // }
            if(!isFast) { // 物体速度慢
                auto detectionWithAny = detectionWithAssociationVariance; // 方便代码阅读
                std::tie(j, error) = getStaticDetectionIndexAndError(predictedPose, object.detection, detectionWithAny); // 根据估计出的物体位姿 T_{l_{k},obj_{i,j}}, 上一时刻物体的检测，找到当前时刻的全部检测中与物体最匹配的检测 (仅提供检测框长宽高信息），并得到对应的检测误差
                if (object.trackScore <= numberOfEarlySteps)
                    error *= 0.2;
                // cout << "slow case: " << "divergence error of " << object.objectIndexForTracking << "th object is equal to " << error << endl;
            } else { // 物体速度快
                if (object.trackScore >= numberOfPreLooseCouplingSteps) {
                    std::tie(j, error) = getDynamicDetectionIndexAndError(predictedPose, detectionWithMatchingVariance); // 误差较大
                } else if (object.trackScore <= numberOfEarlySteps) {
                    std::tie(j, error) = getDynamicDetectionIndexAndError(predictedPose, detectionWithEarlyMatchingVariance); // 因为追踪初期从原物体处继承的速度不准确，容易带来更大的误差，因此需要更宽松的条件使得物体有更多机会参与松耦合图优化
                } else {
                    std::tie(j, error) = getDynamicDetectionIndexAndError(predictedPose, detectionWithMatchingVariance); // 误差较大
                }
                if (object.trackScore <= numberOfEarlySteps)
                    error *= 0.2;
                // cout << "fast case: " << "pose error of " << object.objectIndexForTracking << "th object is equal to " << error << "; dataAssociationError is equal to " << dataAssociationError << endl;
            }
            std::tie(dataAssociationJ, dataAssociationError) = getDynamicDetectionIndexAndError(predictedPose, detectionWithAssociationVariance); // 误差更小，用于确认是否物体框与所有检测框都相差很大

            if (error < matchingError[j]) {
                matchingError[j] = error;
                trackingObjectIndices[j] = -1;
                changeMatchingObject = true;                    
            }

            float detectionMatchThreshold = isFast ? dynamicDetectionMatchThreshold : staticDetectionMatchThreshold;
            // if (error < detectionMatchThreshold && !isWrongDetection(object, detections->boxes[j])) {
            if (error < detectionMatchThreshold) {  // 若物体的检测误差 error 小于设定的阈值 detectionMatchThreshold, 即物体成功匹配到了检测结果
                if (object.lostCount > 0) { // 若物体跟踪已经短暂丢失
                    trackingObjectIndices[j] = object.objectIndexForTracking; // 将物体在跟踪序列中的索引赋值给物体的检测结果对应的 trackingObjectIndices
                    trackingObjectVelocity[j] = object.velocity;                   
                    matchingObjectPtr[j] = &object;
                    matchingObjectPtr[j]->lostCount = std::numeric_limits<int>::max();
                    // cout << object.objectIndexForTracking << "th object is retracked after temporary loss, create a new object for it" << endl;
                    
                } else { // 物体跟踪未丢失
					indicator(i, j)  = 1; // 更新 indicator 矩阵的相应元素为 1, 表示物体 i 和检测 j 成功匹配
					object.lostCount = 0; // 重置 lostCount 为 0，表示该物体当前处于正常跟踪状态
					if (object.trackScore <= numberOfPreLooseCouplingSteps) {
						++object.trackScore; // 增加物体成功跟踪的次数
					}
                    if (trackingObjectIndices[j] > 0) {
                        trackingObjectIndices[j] = -1;
                    }
					object.detection = detections->boxes[j]; // 物体的最优检测结果

                    // ESKF
                    // Eigen::Affine3f trans = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
					// 											      transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // T_{w,l_{i}}
                    // float x, y, z, roll, pitch, yaw;
			        // pcl::getTranslationAndEulerAngles(trans, x, y, z, roll, pitch, yaw);
                    // auto egoTrans = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z)); // T_{w,l_{i}}
                    // auto objectDetectPose = Pose3(Rot3::Quaternion(object.detection.pose.orientation.w, object.detection.pose.orientation.x, object.detection.pose.orientation.y, object.detection.pose.orientation.z),
					// 			                     Point3(object.detection.pose.position.x, object.detection.pose.position.y, object.detection.pose.position.z)); // T_{l_{i},obj_{i,j}}
                    // auto observationPose = egoTrans * objectDetectPose; // T_{w,l_{i}}*T_{l_{i},obj_{i,j}} --> T_{w,obj_{i,j}}
                    // object.updateAndReset(observationPose);
                    // initialEstimate.update(object.poseNodeIndex, object.pose);

                    if (object.trackScore >= numberOfPreLooseCouplingSteps + 1) { // 物体成功跟踪的次数大于等于 numberOfPreLooseCouplingSteps + 1
						auto tightlyCoupledDetectionFactorPtr = boost::make_shared<TightlyCoupledDetectionFactor>(egoPoseKey,
																												  object.poseNodeIndex,
                                                                                                                  object.detection,
                                                                                                                  object.velocity,
																												  detectionWithTightlyCoupledOptimizationVariance);
                        // 空间一致性检查: 检查实际检测位姿和 predictedPose 是否足够接近
						auto spatialConsistencyTest  = error <= 0.2 * detectionMatchThreshold; // error 是否小于等于 0.4*detectionMatchThreshold
                        // 时间一致性检查: 检查物体的速度是否在一段时间内保持一致
						auto temporalConsistencyTest = object.velocityIsConsistent(numberOfVelocityConsistencySteps+1, // 采样数 4
                                                                                   isamCurrentEstimate, // 历史优化结果
																				   objectAngularVelocityConsistencyVarianceThreshold, // 1e-5 线速度协方差
																				   objectLinearVelocityConsistencyVarianceThreshold); // 1e-2 角速度协方差

						if (spatialConsistencyTest && temporalConsistencyTest && error < 0.1) { // 若通过空间一致性检查和时间一致性检查，对物体采用紧耦合
							++numberOfTightlyCoupledObjectsAtThisMoment; // 增加当前时刻紧耦合的物体数量
							anyObjectIsTightlyCoupled = true; // 将 anyObjectIsTightlyCoupled 标记为 true, 表示至少有一个物体被紧密耦合
							object.isTightlyCoupled   = true; // 标记物体和自车之间为紧耦合
							gtSAMgraph.add(LooselyCoupledDetectionFactor(egoPoseKey,
																		 object.poseNodeIndex,
                                                                         object.detection,
                                                                         object.velocity,
																		 detectionWithTightlyCoupledOptimizationVariance,
																		 j)); // 将紧耦合因子添加到 gtSAMgraph 中
							// 保存紧耦合的相关信息
                            object.tightlyCoupledDetectionFactorPtr = tightlyCoupledDetectionFactorPtr;
                            cout << object.objectIndexForTracking << "th object is tracked successfully, everything is fine, use it in tightly coupled graph optimization" << endl; 
						} else { // 若未通过空间一致性检查或时间一致性检查，对物体采用松耦合 paper: Q_{3,1} or Q_{3,2} is not met
							object.isTightlyCoupled = false;
							// 将物体的位姿和速度插入到 initialEstimateForLooselyCoupledObjects 中，准备进行松耦合优化
                            initialEstimateForLooselyCoupledObjects.insert(object.poseNodeIndex, initialEstimate.at(object.poseNodeIndex));
							initialEstimateForLooselyCoupledObjects.insert(object.velocityNodeIndex, initialEstimate.at(object.velocityNodeIndex));
                            // 从 initialEstimate 中移除该物体的位姿和速度，避免其参与紧耦合的优化
							initialEstimate.erase(object.poseNodeIndex);
							initialEstimate.erase(object.velocityNodeIndex);
							gtSAMgraphForLooselyCoupledObjects.add(LooselyCoupledDetectionFactor(egoPoseKey,
																								 object.poseNodeIndex,
                                                                                                 object.detection,
                                                                                                 object.velocity,
																								 detectionWithLooselyCoupledOptimizationVariance,
																								 j)); // 将松耦合因子添加到 gtSAMgraphForLooselyCoupledObjects 中
							// 保存松耦合的相关信息
                            object.looselyCoupledDetectionFactorPtr = boost::make_shared<LooselyCoupledDetectionFactor>(egoPoseKey,
																												        object.poseNodeIndex,
                                                                                                                        object.detection,
                                                                                                                        object.velocity,
																														detectionWithLooselyCoupledOptimizationVariance);
                            // cout << object.objectIndexForTracking << "th object is tracked successfully, fail to pass spatial consistency test or temporal consistency test, use it in loosely coupled graph optimization" << endl;
                        }
					} else { // 物体成功跟踪的次数小于 numberOfPreLooseCouplingSteps+1, 仍对物体采用松耦合
                        object.isTightlyCoupled = false;
                        initialEstimateForLooselyCoupledObjects.insert(object.poseNodeIndex, initialEstimate.at(object.poseNodeIndex));
                        initialEstimate.erase(object.poseNodeIndex);
                        initialEstimateForLooselyCoupledObjects.insert(object.velocityNodeIndex, initialEstimate.at(object.velocityNodeIndex));
						initialEstimate.erase(object.velocityNodeIndex);
						gtSAMgraphForLooselyCoupledObjects.add(LooselyCoupledDetectionFactor(egoPoseKey,
																							 object.poseNodeIndex,
                                                                                             object.detection,
                                                                                             object.velocity,
																							 detectionWithLooselyCoupledOptimizationVariance,
																							 j));
						object.looselyCoupledDetectionFactorPtr = boost::make_shared<LooselyCoupledDetectionFactor>(egoPoseKey,
																													object.poseNodeIndex,
                                                                                                                    object.detection,
                                                                                                                    object.velocity,
																													detectionWithLooselyCoupledOptimizationVariance);
                        // cout << object.objectIndexForTracking << "th object is tracked successfully, but track score is low, use it in loosely coupled graph optimization" << endl;
                    }
				}
			} else {  // 物体的检测误差 error 大于等于 detectionMatchThreshold
                // 注释这段代码是因为假设不会出现 k 时刻仅 A 物体检测到，k + 1 时刻仅与 A 物体相邻的 B 物体检测到的情形
                // if (!isFast && object.trackScore <= numberOfEarlySteps)
                //     error /= 0.4;
                ++object.lostCount; // 物体跟踪正在短暂丢失
                object.trackScore = 0.0;
                PointType objectCenterPoint;
                objectCenterPoint.x = predictedPose.translation().x();
                objectCenterPoint.y = predictedPose.translation().y();
                objectCenterPoint.z = predictedPose.translation().z();
                if ((!isFast && (error < 10 * staticDetectionMatchThreshold)) || isInBoundingBox(objectCenterPoint, detections->boxes[j]) || (isFast && dataAssociationError < dynamicDetectionMatchThreshold)) { // detectionMatchThreshold <= error < 2*detectionMatchThreshold 匹配误差并不过大且物体的中心点在第 j 个检测中; paper: Q_{1} is met
                    if (changeMatchingObject) {
                        trackingObjectIndices[j] = object.objectIndexForTracking;
                        trackingObjectVelocity[j] = object.velocity;
                        matchingObjectPtr[j] = &object;
                    }
                } else { 
                    // cout << object.objectIndexForTracking << "th object's error = " <<  error << " is very high, preserve the object, accumulate lostCount and create an object with new tracking index for it" << endl;
                }
            }            
			++i;
		}
        
        for (int j = 0; j < detections->boxes.size(); j++) {
            if (trackingObjectIndices[j] > 0) {
                if (matchingObjectPtr[j]->lostCount <= trackingStepsForLostObject) {
                    // cout << matchingObjectPtr[j]->objectIndexForTracking << "th object error is a litte high but can be tolerate, so it is temporarily lost, create a new object for it" << endl;
                }
                matchingObjectPtr[j]->lostCount = std::numeric_limits<int>::max();
            }
        }

#ifdef MAP_OPTIMIZATION_DEBUG
		cout << "\n\n"
			 << "INDICATOR ::\n"
			 << indicator << "\n\n";
#endif
        auto detectionWithAny = detectionWithAssociationVariance; // 方便代码阅读
		for (size_t idx = 0; idx < detectionWithAny.size(); ++idx) { // 遍历检测结果
			if (indicator.col(idx).sum() == 0) { // 包含情况: 1、第 idx 个检测有对应的物体，但物体跟踪已经短暂丢失; 2、第 idx 个检测没有对应的物体 ==> 根据检测创建新物体
				ObjectStateESKF object; // 由 idx 个检测生成的物体
				object.detection         = detections->boxes[idx];
				object.pose              = egoPose * detectionWithAny[idx].getPose(); // T_{w,l_{k}}*T_{l_{k},obj} --> T_{w,obj}
				object.velocity          = trackingObjectVelocity[idx]; // 继承原物体速度// object.velocity = Pose3::identity();
				object.poseNodeIndex     = numberOfNodes++;
				object.velocityNodeIndex = numberOfNodes++;
				object.isTightlyCoupled  = false;
                object.objectIndex       = numberOfRegisteredObjects++; // 只要第 idx 个检测有缺陷，就增加 numberOfRegisteredObjects
				object.isFirst           = true; // 新物体标志
				object.timestamp         = timeLaserInfoStamp;
                object.buildNoise();
				if (trackingObjectIndices[idx] < 0) { // 包含情况: 第 idx 个检测没有对应的物体，且匹配误差很大 ==> 创建新跟踪物体 marker
					object.objectIndexForTracking = numberOfTrackingObjects++; // 只有在确认第 idx 个检测无法与任何现有物体相匹配时，才会增加 numberOfTrackingObjects
					// 路径标记
                    visualization_msgs::Marker marker; // visualization_msgs::Marker 是 ROS 中用于可视化的消息类型，用于在 RViz 中显示物体的路径
					marker.id                 = object.objectIndexForTracking;
					marker.type               = visualization_msgs::Marker::SPHERE_LIST; // 球类型
					std_msgs::ColorRGBA color = jsk_topic_tools::colorCategory20(object.objectIndexForTracking); // colorCategory20 根据输入的索引值循环使用一组 20 种不同的颜色，以确保不同对象在可视化中具有不同的颜色，从而更容易区分
					marker.color.a            = 1.0;
					marker.color.r            = color.r;
					marker.color.g            = color.g;
					marker.color.b            = color.b;
					marker.scale.x            = 0.4; // x 方向直径
					marker.scale.y            = 0.4; // y 方向直径
					marker.scale.z            = 0.4; // z 方向直径
					marker.pose.orientation   = tf::createQuaternionMsgFromYaw(0);
					trackingObjectPaths.markers.push_back(marker);

                    visualization_msgs::Marker labelMarker;
                    labelMarker.id      = object.objectIndexForTracking;
                    labelMarker.type    = visualization_msgs::Marker::TEXT_VIEW_FACING;
                    labelMarker.color.a = 1.0;
                    labelMarker.color.r = color.r;
                    labelMarker.color.g = color.g;
                    labelMarker.color.b = color.b;
                    labelMarker.scale.z = 1.2;
                    labelMarker.text    = "Object " + std::to_string(object.objectIndexForTracking);
                    trackingObjectLabels.markers.push_back(labelMarker);

				} else { // 包含情况: 1 第 idx 个检测有对应的物体，但物体跟踪已经短暂丢失; 2 第 idx 个检测没有对应的物体，但匹配误差并不过大 ==> 继续使用原有 marker
                    object.objectIndexForTracking = trackingObjectIndices[idx]; // 第 idx 个检测结果对应的物体在跟踪序列中的索引
                }

				object.box = detectionWithAny[idx].getBoundingBox(); // 从检测结果中提取物体的边界框信息
                objects.back()[object.objectIndex] = object; // 将根据检测结果生成的新物体添加到当前时间步的物体列表中

				// 设置物体的位姿和速度的初始值
                initialEstimateForLooselyCoupledObjects.insert(object.poseNodeIndex, object.pose);
				initialEstimateForLooselyCoupledObjects.insert(object.velocityNodeIndex, object.velocity);
				gtSAMgraphForLooselyCoupledObjects.add(LooselyCoupledDetectionFactor(egoPoseKey,
																					 object.poseNodeIndex,
                                                                                     object.detection,
                                                                                     object.velocity,
																					 detectionWithLooselyCoupledOptimizationVariance,
																					 idx)); // 在松耦合因子图中添加检测因子
				objects.back()[object.objectIndex].looselyCoupledDetectionFactorPtr = boost::make_shared<LooselyCoupledDetectionFactor>(egoPoseKey,
																																		object.poseNodeIndex,
                                                                                                                                        object.detection,
                                                                                                                                        object.velocity,
																																		detectionWithLooselyCoupledOptimizationVariance);

#ifndef ENABLE_COMPACT_VERSION_OF_FACTOR_GRAPH
				// prior velocity factor (the noise should be large)
				auto noise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, 1e0, 1e8, 1e2, 1e2).finished()); // 先验速度因子噪声
				gtSAMgraphForLooselyCoupledObjects.add(PriorFactor<Pose3>(object.velocityNodeIndex, object.velocity, noise)); // 在松耦合因子图中添加先验速度因子
#endif
			} // 若第 idx 个检测有对应的物体，且物体跟踪未丢失，即检测完全正常，则不为该检测创建新物体，即不改变 objects, 也不改变松耦合因子图 gtSAMgraphForLooselyCoupledObjects
		}
	}
    
    pcl::PointCloud<PointType>::Ptr extractObjectCloud(
    const pcl::PointCloud<PointType>::Ptr& cloud,
    const BoundingBox& box) 
    {
        auto result = boost::make_shared<pcl::PointCloud<PointType>>();
        for (const auto& pt : cloud->points) {
            if (isInBoundingBox(pt, box)) {
                result->push_back(pt);
            }
        }
        return result;
    }

    bool runICP(const pcl::PointCloud<PointType>::Ptr& source,
            const pcl::PointCloud<PointType>::Ptr& target,
            Eigen::Matrix4f& tf_out,
            double& error_out,
            double maxCorrDist = 1.0,
            int maxIter = 50,
            double transEps = 1e-6,
            double fitEps = 1e-6)
    {
        if (!source || !target || source->empty() || target->empty()) {
            ROS_WARN("[runICP] Empty source or target cloud!");
            return false;
        }

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setInputSource(source);
        icp.setInputTarget(target);

        // ICP 参数配置
        icp.setMaxCorrespondenceDistance(maxCorrDist);   // 最大匹配点距离
        icp.setMaximumIterations(maxIter);               // 最大迭代次数
        icp.setTransformationEpsilon(transEps);          // 收敛阈值：两次迭代位姿变化小于此值则停止
        icp.setEuclideanFitnessEpsilon(fitEps);          // 收敛阈值：点到点误差小于此值则停止
        icp.setRANSACIterations(0);                      // 不用 RANSAC，加速

        pcl::PointCloud<PointType> aligned;
        icp.align(aligned);

        if (!icp.hasConverged()) {
            ROS_WARN("[runICP] ICP did not converge.");
            return false;
        }

        tf_out = icp.getFinalTransformation();
        error_out = icp.getFitnessScore(); // 越小越好，一般 <0.5 算可接受

        return true;
    }

    inline gtsam::Pose3 eigenToGtsam(const Eigen::Matrix4f& tf)
    {
        Eigen::Matrix3f R = tf.block<3,3>(0,0);
        Eigen::Vector3f t = tf.block<3,1>(0,3);

        return gtsam::Pose3(
            gtsam::Rot3(R.cast<double>()),
            gtsam::Point3(t.cast<double>())
        );
    }

	void saveKeyFramesAndFactor() {
		bool requiredSaveFrame = saveFrame(); // requiredSaveFrame 表示是否为关键帧。若是关键帧，则 requiredSaveFrame = true; 若非关键帧，则 requiredSaveFrame = false
#ifdef ENABLE_SIMULTANEOUS_LOCALIZATION_AND_TRACKING
		if (requiredSaveFrame) {
			// odom factor
			addOdomFactor();

			// gps factor
			addGPSFactor();

			// loop factor
			addLoopFactor();
		} else {
#ifdef ENABLE_ASYNCHRONOUS_STATE_ESTIMATE_FOR_SLOT
			auto egoPose6D      = cloudKeyPoses6D->back(); // 最新关键帧的自车位姿
			Pose3 latestEgoPose = Pose3(Rot3::RzRyRx((Vector3() << egoPose6D.roll, egoPose6D.pitch, egoPose6D.yaw).finished()),
										Point3((Vector3() << egoPose6D.x, egoPose6D.y, egoPose6D.z).finished()));
			initialEstimate.insert(keyPoseIndices.back(), latestEgoPose);
#else
			return;
#endif
		}
#else   // LIO-SAM
		if (!requiredSaveFrame)
			return;

		// odom factor
		addOdomFactor();

		// gps factor
		addGPSFactor();

		// loop factor
		addLoopFactor();
#endif

#ifdef ENABLE_SIMULTANEOUS_LOCALIZATION_AND_TRACKING
		// perform dynamic object propagation
		propagateObjectPoses();

		double tBeforeAddDetectionFactor = ros::Time::now().toSec();
        // detection factor (for multi-object tracking tracking)
		addDetectionFactor(!requiredSaveFrame); // 若是关键帧，则不需要模拟检测，否则需要模拟检测
        double tAfterAddDetectionFactor = ros::Time::now().toSec();

		// constant velocity factor (for multi-object tracking)
		addConstantVelocityFactor();

		// stable pose factor (for multi-object tracking)
		addStablePoseFactor();
        // cout << "number of factors in current factor graph is " << gtSAMgraph.size() << endl;
#endif

#ifdef MAP_OPTIMIZATION_DEBUG
		std::cout << "****************************************************" << endl;
		gtSAMgraph.print("GTSAM Graph:\n");
#endif

		if (!requiredSaveFrame) { // 若非关键帧，则从 initialEstimate 中移除最新的自车位姿
			initialEstimate.erase(keyPoseIndices.back());
		}

		double tBeforeTightlyCoupledOptimization = ros::Time::now().toSec();
        // update iSAM
		isam->update(gtSAMgraph, initialEstimate); // 输入因子图和初始值，执行优化
		isam->update(); // 再次执行优化
        double tAfterTightlyCoupledOptimization = ros::Time::now().toSec();

		if (aLoopIsClosed == true) { // 若存在回环或 GPS 因子，执行多次优化
			isam->update();
			isam->update();
			isam->update();
			isam->update();
			isam->update();
		}
		gtSAMgraph.resize(0); // 清空因子图中的因子(优化变量不会被清空，仍保存在 isam 中)
		initialEstimate.clear(); // 移除当前时间步的优化变量的初始值

#ifdef ENABLE_SIMULTANEOUS_LOCALIZATION_AND_TRACKING
        double tBeforeLooselyCoupledOptimization = ros::Time::now().toSec(); 
		if (!gtSAMgraphForLooselyCoupledObjects.empty()) {
			isam->update(gtSAMgraphForLooselyCoupledObjects, initialEstimateForLooselyCoupledObjects); // 松耦合因子图优化
			isam->update();
		}
        double tAfterLooselyCoupledOptimization = ros::Time::now().toSec();

		gtSAMgraphForLooselyCoupledObjects.resize(0);
		initialEstimateForLooselyCoupledObjects.clear();
#endif

		isamCurrentEstimate = isam->calculateEstimate(); // 优化结果，包括从开始时刻的所有变量

        // 保存物体信息
		if (objects.size() > 0) {
			size_t i = 0;
			for (auto& pairedObject : objects.back()) { // 循环将优化结果赋值给 objects 中的每个物体
				auto& object = pairedObject.second;

				if (object.lostCount > 0) continue; // 当前帧有效物体 lostCount > 0

#ifdef MAP_OPTIMIZATION_DEBUG // FALSE
				std::cout << "(OBJECT " << object.objectIndex << ") [BEFORE]\nPOSITION ::\n"
						  << object.pose << "\n"
						  << "VELOCITY ::\n"
						  << object.velocity << "\n";
#endif

				object.pose = isamCurrentEstimate.at<Pose3>(object.poseNodeIndex); // 更新对象的位姿为优化结果 T_{w,obj}
#ifdef ENABLE_COMPACT_VERSION_OF_FACTOR_GRAPH // FALSE
				if (!object.isFirst) {
					object.velocity = isamCurrentEstimate.at<Pose3>(object.velocityNodeIndex);
				}
#else
				object.velocity = isamCurrentEstimate.at<Pose3>(object.velocityNodeIndex); // 更新对象的速度为优化结果
#endif

#ifdef MAP_OPTIMIZATION_DEBUG
				std::cout << "(OBJECT " << object.objectIndex << ") [AFTER ]\nPOSITION ::\n"
						  << object.pose << "\n"
						  << "VELOCITY ::\n"
						  << object.velocity << "\n\n";
#endif

				auto p                     = object.pose.translation();
				object.box.pose.position.x = p.x();
				object.box.pose.position.y = p.y();
				object.box.pose.position.z = p.z();

				auto r                      = object.pose.rotation();
				object.box.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(r.roll(), r.pitch(), r.yaw());

				object.box.header.frame_id = odometryFrame;
				object.box.label           = object.objectIndex; // 物体的索引

				++i;
			}
		}

		if (requiredSaveFrame) { // 若为关键帧
			// save key poses
			PointType thisPose3D;
			PointTypePose thisPose6D;
			Pose3 latestEstimate;

			latestEstimate = isamCurrentEstimate.at<Pose3>(keyPoseIndices.back()); // 优化得到的最后一个关键帧的位姿
			// cout << "****************************************************" << endl;
			// isamCurrentEstimate.print("Current estimate: ");

			thisPose3D.x         = latestEstimate.translation().x();
			thisPose3D.y         = latestEstimate.translation().y();
			thisPose3D.z         = latestEstimate.translation().z();
			thisPose3D.intensity = cloudKeyPoses3D->size(); // intensity 值为当前关键帧的 id
			cloudKeyPoses3D->push_back(thisPose3D);

			thisPose6D.x         = thisPose3D.x;
			thisPose6D.y         = thisPose3D.y;
			thisPose6D.z         = thisPose3D.z;
			thisPose6D.intensity = thisPose3D.intensity;
			thisPose6D.roll      = latestEstimate.rotation().roll();
			thisPose6D.pitch     = latestEstimate.rotation().pitch();
			thisPose6D.yaw       = latestEstimate.rotation().yaw();
			thisPose6D.time      = timeLaserInfoCur;
			cloudKeyPoses6D->push_back(thisPose6D);

			// cout << "****************************************************" << endl;
			// cout << "Pose covariance:" << endl;
			// cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
			poseCovariance = isam->marginalCovariance(keyPoseIndices.back()); // 取出优化得到的最后一个关键帧位姿的协方差矩阵，用于下次判断是否要添加 GPS 因子

			// 将优化得到的最后一个关键帧的位姿保存到最新优化位姿 transformTobeMapped 当中
			transformTobeMapped[0] = latestEstimate.rotation().roll();
			transformTobeMapped[1] = latestEstimate.rotation().pitch();
			transformTobeMapped[2] = latestEstimate.rotation().yaw();
			transformTobeMapped[3] = latestEstimate.translation().x();
			transformTobeMapped[4] = latestEstimate.translation().y();
			transformTobeMapped[5] = latestEstimate.translation().z();

			// 保存当前关键帧在当前雷达坐标系下的点云(该做法的优势是当位姿被优化后不必调整点云坐标)
			pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
			pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
			pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame); // 将 laserCloudCornerLastDS 复制到 thisCornerKeyFrame 中
			pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);

            singleFrameFilter(thisCornerKeyFrame, thisSurfKeyFrame); // 关联滤除动态点

			cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
			surfCloudKeyFrames.push_back(thisSurfKeyFrame);

			// 更新关键帧路径用于显示
			updatePath(thisPose6D);
		}
	}
    
    void singleFrameFilter(pcl::shared_ptr<pcl::PointCloud<PointType>>& cornerCloud, 
                           pcl::shared_ptr<pcl::PointCloud<PointType>>& surfCloud) {
        int initNumCorner = cornerCloud->size();
        int initNumSurf = surfCloud->size();
        for (auto pairedObject : objects.back()) {
            auto object = pairedObject.second;
            double velocityScale = gtsam::Pose3::Logmap(object.velocity).norm();
            if(velocityScale < 0.5)
                continue;
            cornerCloud->erase(
                std::remove_if(cornerCloud->begin(), cornerCloud->end(), [&](const PointType& point) { return isInBoundingBox(point, object.detection); }),
                cornerCloud->end());
            surfCloud->erase(
                std::remove_if(surfCloud->begin(), surfCloud->end(), [&](const PointType& point) { return isInBoundingBox(point, object.detection); }),
                surfCloud->end());
        }
        int finalNumCorner = cornerCloud->size();
        int finalNumSurf = surfCloud->size();
        // cout << "number of filtered out corner points is " << initNumCorner - finalNumCorner << endl;
        // cout << "number of filtered out surf points is " << initNumSurf - finalNumSurf << endl;
    }

    bool isInBoundingBox(PointType point, BoundingBox box) {
        auto pose = Pose3(Rot3::Quaternion(box.pose.orientation.w, box.pose.orientation.x, box.pose.orientation.y, box.pose.orientation.z),
					      Point3(box.pose.position.x, box.pose.position.y, box.pose.position.z)).inverse(); // T_{l,obj}^{-1} --> T_{obj,l}
        gtsam::Point3 gtsamPoint(point.x, point.y, point.z); // p_{l}
        gtsam::Point3 transformedPoint = pose.transformFrom(gtsamPoint); // T_{obj,l}*p_{l} --> p_{obj}
        if (abs(transformedPoint.x()) < 0.5 * box.dimensions.x &&
            abs(transformedPoint.y()) < 0.5 * box.dimensions.y &&
            abs(transformedPoint.z()) < 0.5 * box.dimensions.z)
            return true;
        else 
            return false;
    }

	void updatePath(const PointTypePose& pose_in) {
		geometry_msgs::PoseStamped pose_stamped;
		pose_stamped.header.stamp       = ros::Time().fromSec(pose_in.time);
		pose_stamped.header.frame_id    = odometryFrame;
		pose_stamped.pose.position.x    = pose_in.x;
		pose_stamped.pose.position.y    = pose_in.y;
		pose_stamped.pose.position.z    = pose_in.z;
		tf::Quaternion q                = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
		pose_stamped.pose.orientation.x = q.x();
		pose_stamped.pose.orientation.y = q.y();
		pose_stamped.pose.orientation.z = q.z();
		pose_stamped.pose.orientation.w = q.w();

		globalPath.poses.push_back(pose_stamped); // 加入全局轨迹
	}

	void publishOdometry() {
		// Publish odometry for ROS (global)
		nav_msgs::Odometry laserOdometryROS;
		laserOdometryROS.header.stamp          = timeLaserInfoStamp;
		laserOdometryROS.header.frame_id       = odometryFrame;
		laserOdometryROS.child_frame_id        = "odom_mapping";
		laserOdometryROS.pose.pose.position.x  = transformTobeMapped[3];
		laserOdometryROS.pose.pose.position.y  = transformTobeMapped[4];
		laserOdometryROS.pose.pose.position.z  = transformTobeMapped[5];
		laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
		pubLaserOdometryGlobal.publish(laserOdometryROS); // 发布当前关键帧的激光里程计信息，供之后的 IMU 里程计使用

		// 发布激光雷达坐标系到里程计坐标系之间的坐标变换 T_{w,l}
		static tf::TransformBroadcaster br;
		tf::Transform t_odom_to_lidar            = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
																 tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
		// tf::StampedTransform(const tf::Transform& transform, const ros::Time& stamp, const std::string& frame_id, const std::string& child_frame_id);
        // transform: 从子坐标系到父坐标系的位姿变换 T_{frame_id,child_frame_id}
        // stamp: 时间戳
        // frame_id: 父坐标系(目标坐标系)名称
        // child_frame_id: 子坐标系(源坐标系)名称
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        // void tf::TransformBroadcaster::sendTransform(const StampedTransform & transform)
        // 发送一个 StampedTransform 对象，其他节点可以通过订阅这个主题来获取和使用这些变换信息
		br.sendTransform(trans_odom_to_lidar);

		// Publish odometry for ROS (incremental)
		static bool lastIncreOdomPubFlag = false;
		static nav_msgs::Odometry laserOdomIncremental;  // incremental odometry msg
		static Eigen::Affine3f increOdomAffine;          // incremental odometry in affine
		if (lastIncreOdomPubFlag == false) { // 首次进入
			lastIncreOdomPubFlag = true;
			laserOdomIncremental = laserOdometryROS;
			increOdomAffine      = trans2Affine3f(transformTobeMapped); // 将当前位姿作为增量式里程计的上一帧位姿
		} else { // 非首次进入
			Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack; // T_{l_{i-1},l_i}
			increOdomAffine             = increOdomAffine * affineIncre; // 然后将变换加到增量式里程计的上一帧位姿上，得到当前帧的增量式里程计位姿 (T_{w,l_{i-1}}^{-1})*T_{l_{i-1},l_i} --> T_{w,l_i}
			float x, y, z, roll, pitch, yaw;
			pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);
			if (cloudInfo.imuAvailable == true) { // 将 IMU 数据与增量式里程计数据进行数据融合
				if (std::abs(cloudInfo.imuPitchInit) < 1.4) {
					double imuWeight = 0.1;
					tf::Quaternion imuQuaternion;
					tf::Quaternion transformQuaternion;
					double rollMid, pitchMid, yawMid;
					// slerp roll
					transformQuaternion.setRPY(roll, 0, 0);
					imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
					tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
					roll = rollMid;
					// slerp pitch
					transformQuaternion.setRPY(0, pitch, 0);
					imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
					tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
					pitch = pitchMid;
				}
			}
			laserOdomIncremental.header.stamp          = timeLaserInfoStamp;
			laserOdomIncremental.header.frame_id       = odometryFrame;
			laserOdomIncremental.child_frame_id        = "odom_mapping";
			laserOdomIncremental.pose.pose.position.x  = x;
			laserOdomIncremental.pose.pose.position.y  = y;
			laserOdomIncremental.pose.pose.position.z  = z;
			laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
			if (isDegenerate) // 判断是否退化
				laserOdomIncremental.pose.covariance[0] = 1; // 如果发生退化，给当前位姿的计算加一个大的方差
			else
				laserOdomIncremental.pose.covariance[0] = 0;
		}
		pubLaserOdometryIncremental.publish(laserOdomIncremental); // 将增量式里程计数据发布，供 IMUPreintegration 订阅
	}

	void publishFrames() {
		if (cloudKeyPoses3D->points.empty())
			return;
		// sensor_msgs::PointCloud2 publishCloud(ros::Publisher * thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
        // thisPub: 指向 ROS 发布者的指针。发布者负责将消息发布到指定的话题上
        // thisCloud: 指向 PCL 点云的智能指针
        // thisStamp: 时间戳，表示点云数据的时间信息
        // thisFrame: 参考坐标系的名称
        // 该函数将 PCL 中的点云数据发布为 ROS 的 sensor_msgs::PointCloud2 消息
		publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame); // 发布历史关键帧位移集合
		publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame); // 发布局部地图的降采样平面点集合
		// 发布世界坐标系下当前激光帧角点，平面点点云(降采样后)
        if (pubRecentKeyFrame.getNumSubscribers() != 0) {
			pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
			PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
			*cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
			*cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);
			publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
		}
		// 发布世界坐标系下当前帧去畸变后的原始点云
		if (pubCloudRegisteredRaw.getNumSubscribers() != 0) {
			pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
			pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
			PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
			*cloudOut                = *transformPointCloud(cloudOut, &thisPose6D);
			publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
		}
		// 发布路径
		if (pubPath.getNumSubscribers() != 0) {
			globalPath.header.stamp    = timeLaserInfoStamp;
			globalPath.header.frame_id = odometryFrame;
			pubPath.publish(globalPath);
		}
		// 发布检测结果
		if (pubDetection.getNumSubscribers() != 0 && detectionIsActive) {
			pubDetection.publish(detections);
		}
        // 发布自身坐标系下当前帧去畸变后的原始点云
		if (pubLaserCloudDeskewed.getNumSubscribers() != 0) {
			cloudInfo.header.stamp = timeLaserInfoStamp;
			pubLaserCloudDeskewed.publish(cloudInfo.cloud_deskewed);
		}
		// 发布追踪物体信息
		if (detectionIsActive) {
			BoundingBoxArray trackingObjectMessage;

			trackingObjectMessage.header          = detections->header;
			trackingObjectMessage.header.frame_id = odometryFrame;
			trackingObjectMessage.header.stamp    = timeLaserInfoStamp;

			std::vector<bool> trackingObjectIsActive(numberOfTrackingObjects, false); // 记录每个跟踪物体是否处于活动状态。
			for (auto& pairedObject : objects.back()) { // 遍历当前帧检测到的物体
				auto& object = pairedObject.second;
				if (object.lostCount == 0) { // 只有物体和检测匹配完全正常，或为新创建的物体，才会更新路径点
					trackingObjectIsActive[object.objectIndexForTracking] = true; // 更新跟踪状态

					// 更新边界框
					object.box.header.stamp = timeLaserInfoStamp;
					trackingObjectMessage.boxes.push_back(object.box);
					trackingObjectMessage.boxes.back().label = object.objectIndexForTracking;

					// 更新路径
					geometry_msgs::Point point;
					point.x = object.box.pose.position.x;
					point.y = object.box.pose.position.y;
					point.z = object.box.pose.position.z;

					trackingObjectPaths.markers[object.objectIndexForTracking].points.push_back(point);
					trackingObjectPaths.markers[object.objectIndexForTracking].header.frame_id = odometryFrame;
					trackingObjectPaths.markers[object.objectIndexForTracking].header.stamp    = timeLaserInfoStamp;

                    trackingObjectLabels.markers[object.objectIndexForTracking].pose.position.x = object.box.pose.position.x;
                    trackingObjectLabels.markers[object.objectIndexForTracking].pose.position.y = object.box.pose.position.y;
                    trackingObjectLabels.markers[object.objectIndexForTracking].pose.position.z = object.box.pose.position.z + 2.0;
                    trackingObjectLabels.markers[object.objectIndexForTracking].header.frame_id = odometryFrame;
                    trackingObjectLabels.markers[object.objectIndexForTracking].header.stamp    = timeLaserInfoStamp;

				}
			}

			pubTrackingObjects.publish(trackingObjectMessage);
			pubTrackingObjectPaths.publish(trackingObjectPaths);
            pubTrackingObjectLabels.publish(trackingObjectLabels);

            // static int frame_id = 0;
            // saveTrackingResultsKITTI(objects, frame_id);
            // frame_id++;
		}

		lio_track::Diagnosis diagnosis; // 诊断信息
		diagnosis.header.frame_id               = odometryFrame;
		diagnosis.header.stamp                  = timeLaserInfoStamp;
		diagnosis.numberOfDetections            = detections ? detections->boxes.size() : 0;
		diagnosis.computationalTime             = timer.elapsed();
		diagnosis.numberOfTightlyCoupledObjects = numberOfTightlyCoupledObjectsAtThisMoment;
		pubDiagnosis.publish(diagnosis);
	}

    void path_save(nav_msgs::Odometry odomAftMapped ){
	    // 保存轨迹，path_save 是文件目录, txt文件提前建好
        std::ofstream pose1(save_traj_to_path, std::ios::app);
        pose1.setf(std::ios::scientific, std::ios::floatfield);
        pose1.precision(9);

        static double timeStart = odomAftMapped.header.stamp.toSec();
        auto T1 =ros::Time().fromSec(timeStart);
        pose1<< odomAftMapped.header.stamp << " "
            << -odomAftMapped.pose.pose.position.y << " "
            << -odomAftMapped.pose.pose.position.z << " "
            << odomAftMapped.pose.pose.position.x << " "
            << odomAftMapped.pose.pose.orientation.x << " "
            << odomAftMapped.pose.pose.orientation.y << " "
            << odomAftMapped.pose.pose.orientation.z << " "
            << odomAftMapped.pose.pose.orientation.w << std::endl;
        pose1.close();        
    }

    // void saveTrackingResultsKITTI(const std::vector<std::map<uint64_t, ObjectStateESKF>>& objects,
    //                           int frame_id) {
    //     std::ofstream ofs(save_tracking_to_path, std::ios::app);
    //     if (!ofs.is_open()) {
    //         std::cerr << "Failed to open tracking result file: " << save_tracking_to_path << std::endl;
    //         return;
    //     }

    //     // 遍历这一帧所有的物体
    //     for (auto& pairedObject : objects.back()) {
    //         auto& object = pairedObject.second;
    //         if (object.lostCount > 0) continue;  // 跳过丢失的

    //         // KITTI Tracking 格式示例
    //         ofs << frame_id << " "
    //             << object.objectIndexForTracking << " "
    //             << "Car" << " " // 或者根据检测类别填 "Pedestrian"/"Cyclist"
    //             << 0 << " " << -1 << " " << -10 << " " // truncated, occluded, alpha (占位)
    //             << 0 << " " << 0 << " " << 50 << " " << 50 << " "; // 如果没有2D bbox，用占位
    //         ofs << object.box.dimensions.z << " "  // h
    //             << object.box.dimensions.y << " "  // w
    //             << object.box.dimensions.x << " "  // l
    //             << object.box.pose.position.x << " "
    //             << object.box.pose.position.y << " "
    //             << object.box.pose.position.z << " "
    //             << 0; // rotation_y 占位
    //         ofs << std::endl;
    //     }
    // }

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "lio_track");

	mapOptimization MO;

	ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

	std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
	std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

	ros::spin();

	loopthread.join();
	visualizeMapThread.join();

	return 0;
}
