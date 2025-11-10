#include "lio_track/cloud_info.h"
#include "utility.h"

struct VelodynePointXYZIRT {
	PCL_ADD_POINT4D
	PCL_ADD_INTENSITY;
	uint16_t ring;
	float time; 
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePointXYZIRT,
							      (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint16_t, ring, ring)(float, time, time))

struct OusterPointXYZIRT {
	PCL_ADD_POINT4D;
	float intensity;
	uint32_t t;
	uint16_t reflectivity;
	uint8_t ring; 
	uint16_t noise;
	uint32_t range;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
								    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, t, t)(uint16_t, reflectivity, reflectivity)(uint8_t, ring, ring)(uint16_t, noise, noise)(uint32_t, range, range))

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer { 
private:
	std::mutex imuLock; // 所有访问 imuQueue 的地方都要用同一把 imuLock
	std::mutex odoLock;

	ros::Subscriber subLaserCloud;
	ros::Publisher pubLaserCloud;

	ros::Publisher pubExtractedCloud;
	ros::Publisher pubLaserCloudInfo;

	ros::Publisher pubReady;

	ros::Subscriber subImu;
	std::deque<sensor_msgs::Imu> imuQueue;

	ros::Subscriber subOdom;
	std::deque<nav_msgs::Odometry> odomQueue;

	std::deque<sensor_msgs::PointCloud2> cloudQueue;
	sensor_msgs::PointCloud2 currentCloudMsg;

	double *imuTime = new double[queueLength];
	double *imuRotX = new double[queueLength];
	double *imuRotY = new double[queueLength];
	double *imuRotZ = new double[queueLength];

	int imuPointerCur;
	bool firstPointFlag;
	Eigen::Affine3f transStartInverse;

	pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
	pcl::PointCloud<PointXYZIRT>::Ptr rawCloudIn;
	pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
	pcl::PointCloud<PointType>::Ptr rawCloud;
	pcl::PointCloud<PointType>::Ptr fullCloud;
	pcl::PointCloud<PointType>::Ptr extractedCloud;

	int deskewFlag;
	cv::Mat rangeMat;

	bool odomDeskewFlag;
	float odomIncreX;
	float odomIncreY;
	float odomIncreZ;

	lio_track::cloud_info cloudInfo; // 当前激光帧点云相关信息
	double timeScanCur;
	double timeScanEnd;
	std_msgs::Header cloudHeader;

public:
	ImageProjection() : deskewFlag(0) {
        // Subscriber ros::NodeHandle::subscribe<M>(const std::string & topic, uint32_t queue_size, const boost::function<void(const boost::shared_ptr<M const>&)>& callback,-- 
        // --const VoidConstPtr & tracked_object = VoidConstPtr(), const TransportHints & transport_hints = TransportHints())
        // M: 订阅的消息类型
        // topic: 要订阅的话题名称
        // queue_size: 队列中最大可以保存的消息数，超过此阈值，先进的先销毁
        // callback: 回调函数，当每个消息到来时，ROS 会自动调用回调函数来处理这个消息
        // tracked_object: 跟踪的对象。这是一个可选参数，通常用于生命周期管理，确保在对象销毁时自动取消订阅。默认值为 VoidConstPtr()，表示没有跟踪对象
        // transport_hints: 传输选项，用于设置消息传输的协议和选项，常见的选项包括 TCP 和 UDP，以及是否启用 Nagle 算法等
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay()); // 订阅原始 IMU 数据
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic + "_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay()); // 订阅 IMU 里程计信息，由 imuPreintegration 计算得到的每时刻 IMU 位姿
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay()); // 订阅原始的 lidar 数据

        // Publisher ros::NodeHandle::advertise<M>(const std::string & topic, uint32_t queue_size, bool latch = false)
        // M: 发布的消息类型
        // topic: 发布到的话题
        // queue_size: 队列中最大可以保存的消息数，超过此阈值，先进的先销毁
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_track/deskew/cloud_deskewed", 1); // 发布激光帧畸变矫正后的点云
        pubLaserCloudInfo = nh.advertise<lio_track::cloud_info>("lio_track/deskew/cloud_info", 1); // 发布激光帧畸变校正后的点云消息
        pubReady          = nh.advertise<std_msgs::Empty>("lio_track/ready", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // PCL 日志级别，只打印 ERROR 日志
	}

	void allocateMemory() {
	    // 分配和初始化 ImageProjection 类中使用的各种点云数据结构和其他相关变量
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        rawCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        rawCloud.reset(new pcl::PointCloud<PointType>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN * Horizon_SCAN);  // N_SCAN：点云的扫描线数; Horizon_SCAN：单个scan中每条扫描线上的点数

        cloudInfo.startRingIndex.assign(N_SCAN, 0);  // 存储每条扫描线的起始索引
        cloudInfo.endRingIndex.assign(N_SCAN, 0);    // 存储每条扫描线的结束索引

        cloudInfo.pointColInd.assign(N_SCAN * Horizon_SCAN, 0);  // 存储每个点的列索引
        cloudInfo.pointRange.assign(N_SCAN * Horizon_SCAN, 0);   // 存储每个点的距离

        resetParameters();
	}

	void resetParameters() {
        // 需要重置的对象的有: 
        // 当前帧激光雷达 ROS 格式的原始点云 laserCloudIn
        // 去畸变后的 PointType 格式的激光雷达点云 extractedCloud
        // 存储当前帧激光雷达点深度信息的 N_SCAN×Horizon_SCAN 大小的矩阵 rangeMat
        // 当前激光雷达帧对应的 IMU 数据个数
        // 当前激光雷达帧第一个激光雷达点的标志位 firstPointFlag
        // 
        // 对应于当前激光雷达帧的 IMU 数据 imuTime, imuRotX, imuRotY, imuRotZ
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur  = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;
        // 重置 IMU 数据数组
        for (int i = 0; i < queueLength; ++i) {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
	}

	~ImageProjection() {}

	void imuHandler(const sensor_msgs::Imu::ConstPtr &imuMsg) {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg); // 将 IMU 原始测量数据转换到 LiDAR 系

        std::lock_guard<std::mutex> lock1(imuLock); // 在这段代码执行期间，imuLock 被锁定，确保在同一时间内只有一个线程可以访问和修改 imuQueue
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x <<
        //       ", y: " << thisImu.linear_acceleration.y <<
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x <<
        //       ", y: " << thisImu.angular_velocity.y <<
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
	}

	void odometryHandler(const nav_msgs::Odometry::ConstPtr &odometryMsg) {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
	}

	void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {
        // cachePointCloud(laserCloudMsg)：将接收到的点云数据缓存起来。如果缓存失败，返回 false
        // deskewInfo()：获取点云的时间校正信息——当前帧起止时刻对应的 IMU 数据、IMU 里程计数据。如果时间校正信息获取失败，返回 false
        if (!cachePointCloud(laserCloudMsg) || !deskewInfo()) {
            pubReady.publish(std_msgs::Empty());  // 如果缓存点云数据失败或时间校正信息获取失败，则发布一个空消息 std_msgs::Empty 并返回，表示处理失败
            return;
        }

        projectPointCloud();

        cloudExtraction(); // 存储有效激光点

        publishClouds(); // 发布校正后的激光点云

        resetParameters(); // 接收每帧 lidar 数据后都要重置参数
	}

	// 这段代码定义了一个函数 cachePointCloud，用于缓存接收到的激光雷达点云数据 laserCloudMsg 到 cloudQueue，并进行必要的转换和检查
	bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg) {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)  // 如果队列中的消息数量小于等于2，则返回 false，表示当前队列中缓存的雷达帧数量小于等于2
            return false;

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());  // 取出激光点云队列中最早的一帧作为当前帧，std::move 用于将队列中的第一个元素高效地移动到 currentCloudMsg 中，避免不必要的拷贝
        cloudQueue.pop_front(); 
        if (sensor == SensorType::VELODYNE) { // 如果传感器是 Velodyne
            pcl::moveFromROSMsg(currentCloudMsg, *rawCloudIn); // 从 ROS 格式的 currentCloudMsg 转换为 PointXYZIRT 格式的 rawCloudIn
            *laserCloudIn += *rawCloudIn;
            // 清空 rawCloud
            rawCloud->clear();
            // 从 PointXYZIRT 格式的 rawCloudIn 中提取 x,y,z,i 信息填充 rawCloud
            for (const auto &p : *rawCloudIn) {
                PointType xyzi;
                xyzi.x         = p.x;
                xyzi.y         = p.y;
                xyzi.z         = p.z;
                xyzi.intensity = p.intensity;
                rawCloud->push_back(xyzi);
            }

        } else if (sensor == SensorType::OUSTER) { // 如果传感器是Ouster
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++) {
                auto &src     = tmpOusterCloudIn->points[i];
                auto &dst     = laserCloudIn->points[i];
                dst.x         = src.x;
                dst.y         = src.y;
                dst.z         = src.z;
                dst.intensity = src.intensity;
                dst.ring      = src.ring;
                dst.time      = src.t * 1e-9f;
            }

        } else {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        cloudHeader = currentCloudMsg.header; // Header用于包含消息的元数据，如时间戳(为整个帧的第一个点的采集时间)
        timeScanCur = cloudHeader.stamp.toSec(); // 当前帧开始时间
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time; // 当前帧截止时间

        // 检查是否存在无效点
        if (laserCloudIn->is_dense == false) {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // 检查 ring 字段
        static int ringFlag = 0;
        if (ringFlag == 0) { // ringFlag = 0 表示尚未检查过 ring 字段
            ringFlag = -1; // ringFlag = -1 表示未找到ring字段
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i) {
                if (currentCloudMsg.fields[i].name == "ring") {
                    ringFlag = 1; // ringFlag = 1 表示已检查到 ring 字段
                    break;
                }
            }
            if (ringFlag == -1) {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0) {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields) {
                if (field.name == "time" || field.name == "t") {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
            ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
	}

	bool deskewInfo() {
        // 同时锁住 imuLock 和 odoLock，确保在同一时间内只有一个线程可以访问和修改这两个队列
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // 检查 IMU 队列 是否为空，IMU 队列中的最早时间是否晚于当前帧起始时间，IMU 队列中的最晚时间是否早于当前帧结束时间
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd) {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        } // 理想情况：imuQueue.front().header.stamp.toSec() <= timeScanCur < timeScanEnd <= imuQueue.back().header.stamp.toSec()

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
	}

    // 将最接近当前帧雷达起始时刻的 IMU 原始旋转保存到点云信息 cloudInfo.imu{Roll/Pitch/Yaw}Init 当中，在 IMU 里程计开始工作后，供后端优化作为雷达旋转先验值使用
    // 对每个 IMU 帧进行角速度积分，得到每个 IMU 帧相对于起始激光雷达帧时刻姿态的相对变换
	void imuDeskewInfo() {
        cloudInfo.imuAvailable = false;

        // 循环检查 imuQueue 中的数据，移除 imuQueue 时间戳早于 timeScanCur - 0.01 秒的 IMU 消息
        while (!imuQueue.empty()) {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front(); 
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i) {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime       = thisImuMsg.header.stamp.toSec(); // 获取 thisImuMsg 的时间戳

            // 如果当前 IMU 消息的时间戳小于等于 timeScanCur，则将该 IMU 消息的滚转角(roll)、俯仰角(pitch)和偏航角(yaw)赋值给 cloudInfo 结构中的相应字段
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit); 
            // 如果当前 IMU 消息的时间戳大于 timeScanEnd + 0.01，则跳出循环，表示后续的 IMU 消息不再需要处理
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            // 初始化 imuRotX、imuRotY 和 imuRotZ 数组的第一个元素为 0，并记录当前时间戳，然后递增 imuPointerCur
            if (imuPointerCur == 0) {
                // i_begin
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z); 

            double timeDiff        = currentImuTime - imuTime[imuPointerCur - 1]; // 计算当前 IMU 消息与前一个 IMU 消息之间的时间差
            // imuRotX, imuRotY, imuRotZ, imuTime 是对应于当前激光雷达帧的 IMU 数据
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur - 1] + angular_x * timeDiff; // 累加 x 轴上的旋转角度
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur - 1] + angular_y * timeDiff; // 累加 y 轴上的旋转角度
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur - 1] + angular_z * timeDiff; // 累加 z 轴上的旋转角度
            imuTime[imuPointerCur] = currentImuTime; // 记录当前 IMU 消息的时间戳
            ++imuPointerCur; // 指向下一个 IMU 消息
        }

        --imuPointerCur; // 指向最后一个有效处理的 IMU 消息

        if (imuPointerCur <= 0) // 如果 imuPointerCur 小于等于 0，表示没有有效的 IMU 数据，直接返回
            return;

        cloudInfo.imuAvailable = true;
	}

    // 将最接近当前雷达帧起始时刻的里程计位姿保存到点云信息 cloudInfo.initialGuess{X/Y/Z}, cloudInfo.initialGuess{Roll/Pitch/Yaw} 中，在 IMU 里程计开始工作后，供后端优化作为雷达位姿先验值使用
    // 通过计算与雷达帧结束时刻最近和与雷达帧起始时刻最近的 IMU 里程计位姿之间的相对变换，得到雷达帧时间段内的相对变换
	void odomDeskewInfo() {
        cloudInfo.odomAvailable = false; // 初始化 cloudInfo 结构中的 odomAvailable 字段为 false，表示初始状态下没有可用的里程计数据

        // 循环检查 odomQueue 中的数据，移除时间戳早于 timeScanCur - 0.01 秒的里程计消息
        while (!odomQueue.empty()) {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
	    }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        nav_msgs::Odometry startOdomMsg; // 声明一个变量来存储扫描开始时的里程计消息

        // 遍历 odomQueue 中的所有里程计消息，找到时间戳大于 timeScanCur 且最接近 timeScanCur 的消息
        for (int i = 0; i < (int)odomQueue.size(); ++i) {
            startOdomMsg = odomQueue[i];
            if (ROS_TIME(&startOdomMsg) < timeScanCur) 
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation); 

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 从四元数中提取滚转角(roll)、俯仰角(pitch)和偏航角(yaw)

        // 用当前激光帧起始时刻的 IMU 里程计初始化 LiDAR 位姿，后面用于 mapOptmization
        cloudInfo.initialGuessX     = startOdomMsg.pose.pose.position.x; // 将扫描开始时的 x 坐标赋值给 cloudInfo 结构中的 initialGuessX
        cloudInfo.initialGuessY     = startOdomMsg.pose.pose.position.y; // 将扫描开始时的 y 坐标赋值给 cloudInfo 结构中的 initialGuessY
        cloudInfo.initialGuessZ     = startOdomMsg.pose.pose.position.z; // 将扫描开始时的 z 坐标赋值给 cloudInfo 结构中的 initialGuessZ
        cloudInfo.initialGuessRoll  = roll; // 将扫描开始时的滚转角赋值给 cloudInfo 结构中的 initialGuessRoll
        cloudInfo.initialGuessPitch = pitch; // 将扫描开始时的俯仰角赋值给 cloudInfo 结构中的 initialGuessPitch
        cloudInfo.initialGuessYaw   = yaw; // 将扫描开始时的偏航角赋值给 cloudInfo 结构中的 initialGuessYaw

        cloudInfo.odomAvailable = true; // 标记已经获取里程计数据

        // odomDeskewFlag 赋值为 false，表示初始状态下没有完成里程计校准
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd) // 如果最新的里程计消息的时间戳小于 timeScanEnd，则直接返回，表示没有在扫描结束之后的有效里程计数据
            return;

        nav_msgs::Odometry endOdomMsg; // 声明一个变量来存储扫描结束时的里程计消息

        // 遍历 odomQueue 中的所有里程计消息，找到时间戳小于 timeScanEnd 且最接近 timeScanEnd 的消息
        for (int i = 0; i < (int)odomQueue.size(); ++i) {
            endOdomMsg = odomQueue[i];
            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }
        
        // 检查扫描开始和结束时的里程计消息的协方差矩阵是否一致，如果不一致则返回
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // 获取扫描开始时的变换矩阵 
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw); // T_{w,l_begin}
        
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);

        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        // 获取扫描结束时的变换矩阵 
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw); // T_{w,l_end}

        // 计算从扫描开始到扫描结束的相对变换矩阵 
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd; // T_{l_begin,l_end}

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre); // 从相对变换矩阵中提取平移增量和旋转增量

        odomDeskewFlag = true; // 标记里程计校准已完成
	}

    // findRotation 函数用于根据给定的时间点 pointTime，查找对应的 IMU 旋转角度。这些旋转角度用于校准点云数据
	void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur) {
        *rotXCur = 0;
        *rotYCur = 0;
        *rotZCur = 0;

        int imuPointerFront = 0;
        // 使用 imuPointerFront 遍历 imuTime 数组，找到第一个大于 pointTime 的时间点
        while (imuPointerFront < imuPointerCur) {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // 如果 pointTime 大于 imuTime[imuPointerFront] 或者 imuPointerFront 为 0 (pointTime 在 imuTime 之外)，直接使用 imuPointerFront 对应的旋转角度
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else { // 如果 pointTime 在两个相邻的时间点之间，使用线性插值计算旋转角度
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront  = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack   = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur           = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur           = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur           = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur) {
        *posXCur = 0;
        *posYCur = 0;
        *posZCur = 0;
	}

	PointType deskewPoint(PointType *point, double relTime) {
        // 如果 deskewFlag 为 -1 或者 cloudInfo.imuAvailable 为 false，表示不需要进行时间校准，直接返回原始点
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime; // 计算当前点的时间戳 pointTime，它是扫描开始时间 timeScanCur 加上相对时间 relTime

        // 查找 pointTime 对应的相对于初始 IMU 的旋转累积 rotX_{i_begin,t}, ...
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);
        // 查找 pointTime 对应的相对于初始 IMU 的位置累积 posX_{i_begin,t}, ..., 实际为0
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true) { // firstPointFlag == true 表示为当前激光帧的第一个点
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse(); // T_{i_begin,t_0}^{-1}，注意 t_0 时刻的位姿是 i_begin 和 i_{begin+1} 加权的结果
            firstPointFlag    = false;
	    }

        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur); // T_{i_begin,t}
        Eigen::Affine3f transBt    = transStartInverse * transFinal; // T_{i_begin,t_0}^{-1}*T_{i_begin,t} --> T_{t_0,t}

        PointType newPoint;
        newPoint.x         = transBt(0, 0) * point->x + transBt(0, 1) * point->y + transBt(0, 2) * point->z + transBt(0, 3);
        newPoint.y         = transBt(1, 0) * point->x + transBt(1, 1) * point->y + transBt(1, 2) * point->z + transBt(1, 3);
        newPoint.z         = transBt(2, 0) * point->x + transBt(2, 1) * point->y + transBt(2, 2) * point->z + transBt(2, 3);
        newPoint.intensity = point->intensity;

        return newPoint;
	}

    // 将 lidar 帧点云转到初始时刻的 lidar 坐标系下
	void projectPointCloud() {
        int cloudSize = laserCloudIn->points.size();
        // 使用一个 for 循环遍历点云中的每个点 // range image projection
        for (int i = 0; i < cloudSize; ++i) {
            // 创建一个 PointType 类型的变量 thisPoint，并从 laserCloudIn 中复制当前点的坐标和强度信息
            PointType thisPoint;
            thisPoint.x         = laserCloudIn->points[i].x;
            thisPoint.y         = laserCloudIn->points[i].y;
            thisPoint.z         = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange) // 跳过距离过近或过远的点，lidarMinRange = 1; lidarMaxRange = 1000
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 检查 rowIdn 是否满足下采样条件。如果 rowIdn 不能被 downsampleRate 整除，跳过当前点
            if (rowIdn % downsampleRate != 0)
                continue;

            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI; // 计算点的水平角度值。atan2 返回 -pi 到 pi 的弧度值

            static float ang_res_x = 360.0 / float(Horizon_SCAN); // 计算水平分辨率 ang_res_x，即每个列的角度间隔
            // 计算当前点在范围图像中的列索引 columnIdn
            int columnIdn          = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2; // --> [450,2250]
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN; // [0°,180°]:[1799,900]; [0°,-180°]:[0,900]

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN) // 如果 columnIdn 超出有效范围 [0, Horizon_SCAN)，跳过当前点
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index                = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint; // 将去畸变后的点云坐标数据保存在 fullCloud 中
        }
    }

    void cloudExtraction() {
        int count = 0; // 有效激光点数量
        // 循环遍历每一行（每个垂直扫描线）
        for (int i = 0; i < N_SCAN; ++i) {
            cloudInfo.startRingIndex[i] = count - 1 + 5; // 记录每根扫描线正数第 5 个及之后的激光点在一维数组中的索引为该扫描线的起始索引
            // 环遍历每一列（每个水平扫描线）
            for (int j = 0; j < Horizon_SCAN; ++j) {
                if (rangeMat.at<float>(i, j) != FLT_MAX) { // 是否为有效激光点(距离不是过近或过远)
                    cloudInfo.pointColInd[count] = j; // 记录激光点对应的列数(初始化为 0)
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i, j); // 激光点距离(初始化为 FLT_MAX)
                    extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]); // 从 fullCloud 中提取去畸变后点云坐标数据，注意 extractedCloud 中仅包含有效激光点
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count - 1 - 5; // 记录每根扫描线倒数第 5 个及之前的激光点在一维数组中的索引为该扫描线的结束索引
        }
	}

	void publishClouds() {
        cloudInfo.header         = cloudHeader;
        // 为 rawCloud 和 extractedCloud 添加时间戳 cloudHeader.stamp 和坐标系 lidarFrame，通过 pubExtractedCloud 发布，并返回作为 cloudInfo 的属性
        cloudInfo.cloud_raw      = publishCloud(&pubExtractedCloud, rawCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_deskewed = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        // 发布 cloudInfo
        pubLaserCloudInfo.publish(cloudInfo);
	}
};

int main(int argc, char **argv) {
	ros::init(argc, argv, "lio_track");

	ImageProjection IP;

	ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

	ros::MultiThreadedSpinner spinner(3);
	spinner.spin();

	return 0;
}
