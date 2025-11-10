#include "utility.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>

std::queue<std::vector<rosbag::MessageInstance>> patchedInstances;

float base_rate;
std::string bag_filename;
std::string imu_topic;
std::string lidar_topic;
std::string pub_imu_topic;
std::string pub_lidar_topic;

ros::Publisher pubImu;
ros::Publisher pubLiDAR;

ros::Time currentTime;

void playInstances(const std::vector<rosbag::MessageInstance>& instances) {
	for (auto& instance : instances) {
		auto duration = instance.getTime() - currentTime; // 通过计算当前消息的时间戳与上一条消息的时间差
		if (duration.toSec() > 0) {
			ros::WallDuration(duration.toSec() / base_rate).sleep(); // 根据 base_rate 调整播放速度
		}

		// 检查每条消息的类型，如果是 IMU 消息则通过 pubImu 发布，如果是 LiDAR 消息则通过 pubLiDAR 发布
        auto imuMsg = instance.instantiate<sensor_msgs::Imu>();
		if (imuMsg != nullptr) {
			pubImu.publish(imuMsg);
		}
		auto lidarMsg = instance.instantiate<sensor_msgs::PointCloud2>();
		if (lidarMsg != nullptr) {
			pubLiDAR.publish(lidarMsg);
		}

		currentTime = instance.getTime(); // 每次发布消息后，更新 currentTime 为当前消息的时间戳
	}
}

void odometryIsDoneCallback(const std_msgs::EmptyConstPtr& msg) {
    // 负责播放下一个数据块(patch), 并将该块从队列中移除
	playInstances(patchedInstances.front());
	patchedInstances.pop();
	ROS_INFO("Remaining patches: %lu", patchedInstances.size());
}

int main(int argc, char* argv[]) {
	ros::init(argc, argv, "offline_bag_player");
	ros::NodeHandle _nh("~"); // 创建一个私有命名空间的 NodeHandle, 用于访问节点私有参数
	ros::NodeHandle nh; // 创建一个全局命名空间的 NodeHandle, 用于订阅和发布话题
    
	_nh.param<float>("base_rate", base_rate, 1.0);
	_nh.param<std::string>("bag_filename", bag_filename, "");
	_nh.param<std::string>("imu_topic", imu_topic, "/imu_raw"); // /imu_raw
	_nh.param<std::string>("lidar_topic", lidar_topic, "/points_raw"); // /points_raw
	_nh.param<std::string>("pub_imu_topic", pub_imu_topic, "/imu_raw"); // /imu_raw
	_nh.param<std::string>("pub_lidar_topic", pub_lidar_topic, "/points_raw"); // /points_raw

	ros::Subscriber sub = nh.subscribe<std_msgs::Empty>("lio_track/ready", 10, &odometryIsDoneCallback);
 
	pubImu   = nh.advertise<sensor_msgs::Imu>(pub_imu_topic, 2000); // 队列大小为 2000
	pubLiDAR = nh.advertise<sensor_msgs::PointCloud2>(pub_lidar_topic, 1); // 队列大小为 1

	if (bag_filename.empty()) { // 如果 .bag 文件路径为空，输出错误信息并退出程序
		ROS_ERROR("bag_filename is empty");
		return -1;
	}

	// 使用 rosbag::Bag 类打开指定的 .bag 文件，读取模式为只读
	rosbag::Bag bag;
	ROS_INFO("Opening bag file: %s", bag_filename.c_str());
	bag.open(bag_filename, rosbag::bagmode::Read);

	std::vector<std::string> topics;
	topics.push_back(imu_topic);
	topics.push_back(lidar_topic);
	rosbag::View view(bag, rosbag::TopicQuery(topics)); // view 包含 bag 中话题 imu_topic 或 lidar_topic 对应的消息

	std::vector<rosbag::MessageInstance> instances;
	bool timeIsInitialized = false;
	BOOST_FOREACH (rosbag::MessageInstance const m, view) { // 用 BOOST_FOREACH 遍历 .bag 文件中的所有消息实例
		instances.push_back(m); // 将每条消息添加到 instances 向量中
		if (m.getTopic() == lidar_topic) { // 如果遇到 LiDAR 消息
			patchedInstances.push(instances);
			instances.clear();
		}

		if (!timeIsInitialized) { 
			currentTime       = m.getTime(); // 初始化 currentTime 为第一条消息的时间戳
			timeIsInitialized = true;
		}
	}

	ROS_INFO("Pending ...");
	while (pubImu.getNumSubscribers() < 2 || pubLiDAR.getNumSubscribers() < 1) { // 要求 pubImu 至少有 2 个订阅者，pubLiDAR 至少有 1 个订阅者
		ros::spinOnce();
	}

	ROS_INFO("Start playing the bag ...");
	// 播放 patchedInstances 队列中的第一个数据块，并将其从队列中移除
	playInstances(patchedInstances.front());
	patchedInstances.pop();

	while (!patchedInstances.empty() && ros::ok()) { // 在 patchedInstances 队列不为空且 ROS 系统正常运行的情况下，继续处理 ROS 事件循环，等待后续的数据块被播放
		ros::spinOnce();
	}

	bag.close();

	return 0;
}