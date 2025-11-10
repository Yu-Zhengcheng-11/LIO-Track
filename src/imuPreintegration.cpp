#include "utility.h"

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

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

// 定义类型别名
using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

class TransformFusion : public ParamServer {
 public:
	std::mutex mtx;

	ros::Subscriber subImuOdometry;
	ros::Subscriber subLaserOdometry;

	ros::Publisher pubImuOdometry;
	ros::Publisher pubImuPath;

	Eigen::Affine3f lidarOdomAffine;
	Eigen::Affine3f imuOdomAffineFront;
	Eigen::Affine3f imuOdomAffineBack;

	tf::TransformListener tfListener; // tf::TransformListener 是 ROS1 中用于监听和查询 tf 变换的主要工具。它会自动订阅 /tf 主题，并缓存接收到的变换信息。可以通过该类中的 lookupTransform 方法查询两个坐标系之间的变换
	tf::StampedTransform lidar2Baselink;

	double lidarOdomTime = -1;
	deque<nav_msgs::Odometry> imuOdomQueue; 


    TransformFusion() {
		if (lidarFrame != baselinkFrame) { // 如果 lidar 系与 baselink 系不同
			try {
				// bool tf::Transformer::waitForTransform(const std::string & target_frame, const std::string & source_frame, const ros::Time & time, const ros::Duration & timeout, ...)
                // 阻塞当前线程，等待直到查询到 time 下的坐标系变换 T_{target_frame,source_frame}, 如果在 time_out 时间内没有找到有效的变换，函数将返回 false
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0)); // ros::Time(0) 表示最近的可用时间
				// void tf::Transformer::lookupTransform(const std::string & target_frame, const std::string & source_frame, const ros::Time & time, StampedTransform & transform) const
                // 不会阻塞线程，查询两个坐标系之间的变换 T_{target_frame, source_frame} 并存储到 transform
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
			} catch (tf::TransformException ex) {
				ROS_ERROR("%s", ex.what());
			}
		}
        // 订阅激光里程计，来自 mapOptimization
		subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_track/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
		// 订阅 IMU 里程计，来自 IMU 预积分
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &TransformFusion::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布 IMU 里程计，用于 RViz 显示
		pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        // 发布 IMU 里程计轨迹，用于 RViz 显示 
		pubImuPath     = nh.advertise<nav_msgs::Path>("lio_track/imu/path", 1);
	}

	// 将 nav_msgs::Odometry 消息转换为 Eigen::Affine3f 变换矩阵
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom) {
		double x, y, z, roll, pitch, yaw;
		x = odom.pose.pose.position.x;
		y = odom.pose.pose.position.y;
		z = odom.pose.pose.position.z;
		tf::Quaternion orientation;
		tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
		tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
		return pcl::getTransformation(x, y, z, roll, pitch, yaw);
	}

	// 处理接收到的激光雷达里程计消息，更新激光雷达的导航状态 lidarOdomAffine 和时间戳 lidarOdomTime
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
		std::lock_guard<std::mutex> lock(mtx);
		lidarOdomAffine = odom2affine(*odomMsg); // T_{w,l}
		lidarOdomTime = odomMsg->header.stamp.toSec();
	}

	void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
		// static tf
		static tf::TransformBroadcaster tfMap2Odom;
		static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        // tf::StampedTransform::StampedTransform(const tf::Transform & input, const ros::Time & timestamp, 
        //                                        const std::string & frame_id, const std::string & child_frame_id)
        // 将 input 赋值给 T_{frame_id,child_frame_id} 
		tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame)); // mapFrame = "map", odometryFrame = "odom"

		std::lock_guard<std::mutex> lock(mtx);

		imuOdomQueue.push_back(*odomMsg); // 将接收到的 IMU 里程计消息添加到队列 imuOdomQueue 中 <-- T_{w,l_t}

		if (lidarOdomTime == -1) // 如果 lidarOdomTime 为 -1, 表示还没有接收到激光雷达里程计信息，直接返回
			return;
		while (!imuOdomQueue.empty()) {
			if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime) // 移除队列中所有时间戳小于等于 lidarOdomTime 的消息
				imuOdomQueue.pop_front();
			else
				break;
		}

        // 将低频的 LiDAR 里程计结合 IMU 信息转变为高频的 LiDAR 里程计，并发布      
		Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front()); // 初始 IMU 时刻对应的 IMU 里程计位姿 T_{w,l_begin}
		Eigen::Affine3f imuOdomAffineBack  = odom2affine(imuOdomQueue.back()); // 当前时刻 IMU 里程计位姿 T_{w,l_t}
		Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack; // T_{l_begin,l_t}
		Eigen::Affine3f imuOdomAffineLast  = lidarOdomAffine * imuOdomAffineIncre; // T_{w,l_begin}*T_{l_begin,l_t} --> T_{w,l_t}
		float x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);

		nav_msgs::Odometry laserOdometry    = imuOdomQueue.back(); // T_{w,l_t}
        // t^{w}_{w,l_t}
		laserOdometry.pose.pose.position.x  = x;
		laserOdometry.pose.pose.position.y  = y;
		laserOdometry.pose.pose.position.z  = z;
		laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
		pubImuOdometry.publish(laserOdometry); // 发布高频 LiDAR 里程计

		// publish tf
		static tf::TransformBroadcaster tfOdom2BaseLink;
		tf::Transform tCur;
		tf::poseMsgToTF(laserOdometry.pose.pose, tCur); // 将 laserOdometry 消息中的位姿转换为 tf::Transform 对象 tCur T_{w,l_t}
		if (lidarFrame != baselinkFrame)
			tCur = tCur * lidar2Baselink; // T_{w,l_t}*T_{l_t,base_link} --> T_{w,base_link}
		tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame); 
		tfOdom2BaseLink.sendTransform(odom_2_baselink);

		// 发布高频 LiDAR 里程计路径
		static nav_msgs::Path imuPath; // 存储 IMU 里程计路径
		static double last_path_time = -1; // 记录上次更新路径的时间
		double imuTime               = imuOdomQueue.back().header.stamp.toSec();
		if (imuTime - last_path_time > 0.1) { // 每 0.1s 添加一个路径点
			last_path_time = imuTime;
			geometry_msgs::PoseStamped pose_stamped;
			pose_stamped.header.stamp    = imuOdomQueue.back().header.stamp;
			pose_stamped.header.frame_id = odometryFrame; // 里程计坐标系的名称为 "odom"
			pose_stamped.pose            = laserOdometry.pose.pose;
			imuPath.poses.push_back(pose_stamped); // 将新的姿态添加到 imuPath.poses 中
			while (!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0) // 删除路径中时间戳小于 lidarOdomTime - 1.0 的姿态
				imuPath.poses.erase(imuPath.poses.begin());
			if (pubImuPath.getNumSubscribers() != 0) { // 如果 pubImuPath 发布的话题有订阅者，则发布 IMU 里程计路径
				imuPath.header.stamp    = imuOdomQueue.back().header.stamp;
				imuPath.header.frame_id = odometryFrame; // 里程计坐标系的名称为 "odom"
				pubImuPath.publish(imuPath);
			}
		}
	}
};


class IMUPreintegration : public ParamServer {
 public:
	std::mutex mtx;

    // 订阅与发布
	ros::Subscriber subImu;
	ros::Subscriber subOdometry;
	ros::Publisher pubImuOdometry;

	bool systemInitialized = false;
    
    // 噪声协方差
	gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
	gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
	gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
	gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
	gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
	gtsam::Vector noiseModelBetweenBias;

    // 预积分器
	gtsam::PreintegratedImuMeasurements* imuIntegratorOpt_; // 用于因子图优化的与积分器
	gtsam::PreintegratedImuMeasurements* imuIntegratorImu_; // 用于计算优化后 IMU

    // IMU 数据队列
	std::deque<sensor_msgs::Imu> imuQueOpt; // 用于因子图优化
	std::deque<sensor_msgs::Imu> imuQueImu; // 用于优化后进行位姿重新计算 IMU 位姿的预积分器

	gtsam::Pose3 prevPose_; // 构造函数中默认旋转矩阵为单位矩阵，位移为零向量
	gtsam::Vector3 prevVel_;
	gtsam::NavState prevState_; // 构造函数中默认位移，速度为零向量
	gtsam::imuBias::ConstantBias prevBias_;

	gtsam::NavState prevStateOdom;
	gtsam::imuBias::ConstantBias prevBiasOdom;

	bool doneFirstOpt   = false;
	double lastImuT_imu = -1;
	double lastImuT_opt = -1;

    // ISAM2 优化器
	gtsam::ISAM2 optimizer;
	gtsam::NonlinearFactorGraph graphFactors;
	gtsam::Values graphValues;

	const double delta_t = 0;

	int key = 1; // 循环次数

    // IMU-LiDAR 位姿变换
	gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z())); // (I,-t^{l}_{li})
	gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z())); // (I,t^{l}_{li})

	IMUPreintegration() {
        // 订阅原始 IMU 数据，完成预积分前向递推。队列设置足够大，以防止由于因子图优化时间过长导致 IMU 数据丢包
		subImu      = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅激光里程计，来自 mapOptimization，用两帧之间的 IMU 预积分量构建因子图，优化当前帧位姿
		subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_track/mapping/odometry_incremental", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布 IMU 里程计
		pubImuOdometry = nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000); 

		boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity); // 预积分器重力参数
        // 设置 IMU 预积分噪声的协方差矩阵
		p->accelerometerCovariance                       = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);  // acc white noise in continuous
		p->gyroscopeCovariance                           = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);  // gyro white noise in continuous
		p->integrationCovariance                         = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);         // error committed in integrating position from velocities
		gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // 预积分器先验零偏参数
		;  // assume zero initial bias

		// 设置先验噪声的协方差矩阵
        priorPoseNoise        = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished());  // rad,rad,rad,m, m, m
		priorVelNoise         = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);                                                               // m/s
		priorBiasNoise        = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);                                                              // 1e-2 ~ 1e-3 seems to be good
		// 激光里程计 scan-to-map 优化过程中发生退化，则选择一个较大的协方差矩阵
        correctionNoise       = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());     // rad,rad,rad,m, m, m
		correctionNoise2      = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished());                    // rad,rad,rad,m, m, m
		noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        // 初始化预积分器
		imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
	}

    // 重置 ISMA2 优化器
	void resetOptimization() {
		gtsam::ISAM2Params optParameters;
        // 通过合理设置下面两个参数，可以在保证优化精度的同时，控制计算开销。
		optParameters.relinearizeThreshold = 0.1; // 判断是否需要重新线性化的阈值，若解的变化大于该阈值，则触发重新线性化，重新计算雅可比矩阵
		optParameters.relinearizeSkip      = 1; // 控制重新线性化的频率，表示每隔多少次迭代重新线性化一次
		optimizer                          = gtsam::ISAM2(optParameters);

		gtsam::NonlinearFactorGraph newGraphFactors; // NonlinearFactorGraph 是一个包含所有因子(约束条件)的图
		graphFactors = newGraphFactors; // 创建了一个空的因子图，并将其赋值给成员变量 graphFactors

		gtsam::Values NewGraphValues; // Values 是一个包含所有变量及其初始值的容器
		graphValues = NewGraphValues; // 创建了一个空的初始值集合，并将其赋值给成员变量 graphValues
	}

	void resetParams() {
		lastImuT_imu      = -1;
		doneFirstOpt      = false;
		systemInitialized = false;
	}

    // 利用输入的 LiDAR pose 和 IMU 数据进行图优化，获得一个最合理的状态量估计值
	void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
		std::lock_guard<std::mutex> lock(mtx);

		double currentCorrectionTime = ROS_TIME(odomMsg); // 当激光帧时间戳

		if (imuQueOpt.empty())
			return;

        // t^{w}_{wl}, q^{w}_{wl}
		float p_x              = odomMsg->pose.pose.position.x;
		float p_y              = odomMsg->pose.pose.position.y;
		float p_z              = odomMsg->pose.pose.position.z;
		float r_x              = odomMsg->pose.pose.orientation.x;
		float r_y              = odomMsg->pose.pose.orientation.y;
		float r_z              = odomMsg->pose.pose.orientation.z;
		float r_w              = odomMsg->pose.pose.orientation.w;
		bool degenerate        = (int)odomMsg->pose.covariance[0] == 1 ? true : false; // 是否退化，1 表示退化，0 表示未退化
		gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z)); // T_{wl}

		// initialize system
		if (systemInitialized == false) {
			resetOptimization(); // 重置因子图
 
			while (!imuQueOpt.empty()) {
				if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t) { // delta_t = 0
					lastImuT_opt = ROS_TIME(&imuQueOpt.front());
					imuQueOpt.pop_front(); 
				} else
					break;
			}
			// 添加里程计先验因子
			prevPose_ = lidarPose.compose(lidar2Imu); // 先验位姿 T_{wl}*(I,t^{l}_{li}) --> (R_{wl},t^{w}_{wi})
			gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise); // X(0) 对应优化器内部第 0 个位姿变量。在优化过程中，优化器会尽量使此变量的估计值接近 prevPose_，但也会考虑其他因子的影响
			graphFactors.add(priorPose); // 向因子图中添加先验因子
			// 添加速度先验因子
			prevVel_ = gtsam::Vector3(0, 0, 0);
			gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
			graphFactors.add(priorVel);
			// 添加偏差先验因子
			prevBias_ = gtsam::imuBias::ConstantBias(); // 将 prevBias_ 初始化为零偏置，即认为初始的加速度计和陀螺仪偏置为零
			gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise); // B(0) 表示第 0 个偏置变量
			graphFactors.add(priorBias);
            // 设置优化变量初始值
			graphValues.insert(X(0), prevPose_); // 这个插入操作将 prevPose_ 作为 X(0) 对应变量的初始值添加到 graphValues 中，初始值会影响优化的起点，但不会作为一个硬性约束
			graphValues.insert(V(0), prevVel_);
			graphValues.insert(B(0), prevBias_);
			// 优化一次
			optimizer.update(graphFactors, graphValues);
			graphFactors.resize(0);
			graphValues.clear();
            // 调用 resetIntegrationAndSetBias 方法会重置预积分状态并设置新的零偏
            // 重置预积分状态: 将切空间内 9 维的预积分量设置为 0，同时设置预积分累积时间为 0，切空间预积分量的协方差矩阵为零矩阵，并将预积分器的偏差设置为传入的参数
			imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
			imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

			key               = 1;
			systemInitialized = true; // 标记已经完成初始化
			return;
		}

		// 随着时间的运行，图的规模越来越大，内存和时间会越来越大，所以每 100 帧就把优化器重置一次。过程和初始化很类似，但状态量是前一帧的数据，噪声协方差是边缘化之后留下来的
		if (key == 100) {
			// 边缘化协方差矩阵，即从优化后的状态向量中提取某个特定变量的边际协方差矩阵
			gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
			gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
			gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
			// 重置因子图
			resetOptimization();
			// 添加里程计先验因子
			gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
			graphFactors.add(priorPose);
			// 添加速度先验因子
			gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
			graphFactors.add(priorVel);
			// 添加偏差先验因子
			gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
			graphFactors.add(priorBias);
			// 设置优化变量初始值
			graphValues.insert(X(0), prevPose_);
			graphValues.insert(V(0), prevVel_);
			graphValues.insert(B(0), prevBias_);
			// 优化一次
			optimizer.update(graphFactors, graphValues);
			graphFactors.resize(0);
			graphValues.clear();

			key = 1; // 更新循环标记
		}

		// 通过用于优化的预积分器 imuIntegratorOpt_ 计算前一帧与当前帧之间的 IMU 预积分量
		while (!imuQueOpt.empty()) {
			sensor_msgs::Imu* thisImu = &imuQueOpt.front();
			double imuTime            = ROS_TIME(thisImu);
			if (imuTime < currentCorrectionTime - delta_t) {
				double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
				imuIntegratorOpt_->integrateMeasurement(
						gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
						gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);

				lastImuT_opt = imuTime;
				imuQueOpt.pop_front();
			} else
				break;
		}

        // 添加 IMU 因子
		const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
		gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu); // 关联节点：前一帧位姿，前一帧速度，当前帧位姿，当前帧速度，前一帧偏置，用于优化的预积分器
		graphFactors.add(imu_factor);
		// 在因子图中添加偏差因子
        // gtsam::imuBias::ConstantBias(): 表示前后两帧偏差之间的先验关系为零变化
        // sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias: 上次重置预积分器到当前的时间间隔的平方根乘以 noiseModelBetweenBias 向量
		graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), 
                                                                            gtsam::imuBias::ConstantBias(), gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
		// 添加位姿先验因子
		gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu); // 先验位姿 T_{wl}*(I,t^{l}_{li}) --> (R_{wl},t^{w}_{wi})
		gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
		graphFactors.add(pose_factor);
		// 基于偏置 prevBias_ 修正用于优化的预积分器的预积分值，并根据修正后的预积分值结合上一激光雷达帧位姿 prevState_ 预测新的导航状态(位姿+速度)
		gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        // 用估计出的状态为变量节点赋初值
		graphValues.insert(X(key), propState_.pose());
		graphValues.insert(V(key), propState_.v());
		graphValues.insert(B(key), prevBias_); // 上个激光帧优化出的偏置作为当前激光帧偏置的初值

		optimizer.update(graphFactors, graphValues); // 将新的因子和初始值添加到优化器中，并执行优化，优化过程会改变 X(...),V(...),B(...) 对应的优化器内部变量的值
		optimizer.update(); // 再次触发优化
		graphFactors.resize(0);
		graphValues.clear();
		// 根据预积分优化后的结果更新变量值
		gtsam::Values result = optimizer.calculateEstimate(); // 将优化结果存入 result
        // 更新下列变量的值
		prevPose_            = result.at<gtsam::Pose3>(X(key)); 
		prevVel_             = result.at<gtsam::Vector3>(V(key));
		prevState_           = gtsam::NavState(prevPose_, prevVel_);
		prevBias_            = result.at<gtsam::imuBias::ConstantBias>(B(key)); // 当前激光帧偏置

		imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_); // 重置用于优化的预积分器 imuIntegratorOpt_，清除之前的预积分结果
		// 检查优化
		if (failureDetection(prevVel_, prevBias_)) {
			resetParams();
			return;
		}

		prevStateOdom = prevState_; // 激光里程计计算出的当前帧的位姿，速度
		prevBiasOdom  = prevBias_; // 激光里程计计算出的偏置
		// 从 IMU 队列中删除当前激光里程计时刻之前的数据
		double lastImuQT = -1;
		while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t) {
			lastImuQT = ROS_TIME(&imuQueImu.front());
			imuQueImu.pop_front();
		}
        // 通过用于计算的预积分器 imuIntegratorImu_ 计算从当前激光帧开始的 IMU 预积分量
		if (!imuQueImu.empty()) {
			imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom); // 设置用于计算的预积分器的偏置
			for (int i = 0; i < (int)imuQueImu.size(); ++i) {
				sensor_msgs::Imu* thisImu = &imuQueImu[i];
				double imuTime            = ROS_TIME(thisImu);
				double dt                 = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);  
				imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
														gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt);
				lastImuQT = imuTime;
			}
		}
		++key;
		doneFirstOpt = true;
	}

    // 速度或者偏置过大，认为失败
	bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur) {
		Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
		if (vel.norm() > 30) {
			ROS_WARN("Large velocity, reset IMU-preintegration!");
			return true;
		}
		Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
		Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
		if (ba.norm() > 1.0 || bg.norm() > 1.0) {
			ROS_WARN("Large bias, reset IMU-preintegration!");
			return true;
		}
		return false;
	}

	void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw) {
		std::lock_guard<std::mutex> lock(mtx);

		sensor_msgs::Imu thisImu = imuConverter(*imu_raw); // 将 IMU 原始测量数据转换到 LiDAR 坐标系
		imuQueOpt.push_back(thisImu);
		imuQueImu.push_back(thisImu);

		if (doneFirstOpt == false) // 如果没有进行过基于雷达里程计的因子图优化，则直接退出回调函数
			return;

		double imuTime = ROS_TIME(&thisImu); // 获取当前 IMU 数据的时间戳
		double dt      = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu); // 计算与上一次 IMU 数据的时间间隔 dt，如果 lastImuT_imu 小于0，表示这是第一次接收 IMU 数据，使用默认的时间间隔 1.0 / 500.0
		lastImuT_imu   = imuTime; // 更新 lastImuT_imu

        // 通过新的 IMU 测量数据和时间间隔进行前向传播，更新：1.预积分器的预积分值; 2.预积器的累积时间; 3.切空间预积分量的协方差矩阵
		imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
												gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt);
        // NavState predict(const NavState& state_i, const imuBias::ConstantBias& bias_i, ...) const;
        // 基于偏置 prevBiasOdom 修正用于优化的预积分器的预积分值，并根据修正后的预积分值结合上一激光雷达帧的位姿 prevStateOdom 预测新的导航状态
		gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom); // (R_{wl},t^{w}_{wi})

		// 发布里程计消息
		nav_msgs::Odometry odometry;
		odometry.header.stamp    = thisImu.header.stamp;
		odometry.header.frame_id = odometryFrame; // 坐标信息的参考系, 表示里程计数据是在 odom 坐标系下给出的位置和方向信息
		odometry.child_frame_id  = "odom_imu"; // 里程计所追踪的物体或坐标系，通常是机器人本身的基座 (base_link)，表明里程计数据描述的是 child_frame_id 相对于 frame_id 的运动

		gtsam::Pose3 imuPose   = gtsam::Pose3(currentState.quaternion(), currentState.position()); // (R_{wl},t^{w}_{wi})
		gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar); // (R_{wl},t^{w}_{wi})*(I,-t^{l}_{li}) --> (R_{wl},t^{w}_{wl})

		odometry.pose.pose.position.x    = lidarPose.translation().x();
		odometry.pose.pose.position.y    = lidarPose.translation().y();
		odometry.pose.pose.position.z    = lidarPose.translation().z();
		odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
		odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
		odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
		odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        // v^{w}_{wi}
        odometry.twist.twist.linear.x  = currentState.velocity().x();
		odometry.twist.twist.linear.y  = currentState.velocity().y();
		odometry.twist.twist.linear.z  = currentState.velocity().z();
		odometry.twist.twist.angular.x = thisImu.angular_velocity.x - prevBiasOdom.gyroscope().x();
		odometry.twist.twist.angular.y = thisImu.angular_velocity.y - prevBiasOdom.gyroscope().y();
		odometry.twist.twist.angular.z = thisImu.angular_velocity.z - prevBiasOdom.gyroscope().z();
        // std::cout << "publish message of topic odomerty/imu_incremental, stamp: " << thisImu.header.stamp << std::endl;
		pubImuOdometry.publish(odometry); 
	}
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "roboat_loam");

	IMUPreintegration ImuP;

	TransformFusion TF;

	ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

	ros::MultiThreadedSpinner spinner(4); // 创建线程池
	spinner.spin();

	return 0;
}
