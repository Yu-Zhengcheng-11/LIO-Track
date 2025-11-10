#include "lio_track/cloud_info.h"
#include "utility.h"

struct smoothness_t {
	float value; // 曲率值
	size_t ind; // 激光点一维索引
};

struct by_value {
	bool operator()(smoothness_t const &left, smoothness_t const &right) {
	return left.value < right.value;
	}
};

class FeatureExtraction : public ParamServer {
public:
	ros::Subscriber subLaserCloudInfo;

	ros::Publisher pubLaserCloudInfo;
	ros::Publisher pubCornerPoints;
	ros::Publisher pubSurfacePoints;

	pcl::PointCloud<PointType>::Ptr extractedCloud; 
	pcl::PointCloud<PointType>::Ptr cornerCloud;
	pcl::PointCloud<PointType>::Ptr surfaceCloud;

	pcl::VoxelGrid<PointType> downSizeFilter;

	lio_track::cloud_info cloudInfo;
	std_msgs::Header cloudHeader;

	std::vector<smoothness_t> cloudSmoothness; // 存储有效点云中每个点的曲率
	float *cloudCurvature;
	int *cloudNeighborPicked;
	int *cloudLabel;

	FeatureExtraction() {
        subLaserCloudInfo = nh.subscribe<lio_track::cloud_info>("lio_track/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay()); // 订阅激光帧畸变校正后的点云消息

        pubLaserCloudInfo = nh.advertise<lio_track::cloud_info>("lio_track/feature/cloud_info", 1);
        pubCornerPoints   = nh.advertise<sensor_msgs::PointCloud2>("lio_track/feature/cloud_corner", 1); // 发布角点(边缘点)点云
        pubSurfacePoints  = nh.advertise<sensor_msgs::PointCloud2>("lio_track/feature/cloud_surface", 1); // 发布平面点点云

        initializationValue();
	}

	void initializationValue() {
        cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize); // odometrySurfLeafSize = 0.2

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature      = new float[N_SCAN * Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN]; // 特征提取标记，1 表示遮挡、平行，或者已经进行特征提取的点，0 表示还未进行特征提取处理
        cloudLabel          = new int[N_SCAN * Horizon_SCAN];
	}

	void laserCloudInfoHandler(const lio_track::cloud_infoConstPtr &msgIn) {
        cloudInfo   = *msgIn;                                     // 用 imageProjection 发布的 ROS 格式的校正后的点云数据初始化 cloudInfo 
        cloudHeader = msgIn->header;                              // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud);  // new cloud for extraction

        calculateSmoothness(); // 计算点云中每个点的曲率

        markOccludedPoints(); // 标记遮挡点和平行点

        extractFeatures();

        publishFeatureCloud();
	}

	void calculateSmoothness() {
        int cloudSize = extractedCloud->points.size(); // 有效激光点数量
        for (int i = 5; i < cloudSize - 5; i++) {
            float diffRange = cloudInfo.pointRange[i - 5] + 
                              cloudInfo.pointRange[i - 4] + 
                              cloudInfo.pointRange[i - 3] + 
                              cloudInfo.pointRange[i - 2] + 
                              cloudInfo.pointRange[i - 1] - 
                              cloudInfo.pointRange[i] * 10 + 
                              cloudInfo.pointRange[i + 1] + 
                              cloudInfo.pointRange[i + 2] + 
                              cloudInfo.pointRange[i + 3] + 
                              cloudInfo.pointRange[i + 4] + 
                              cloudInfo.pointRange[i + 5];

            cloudCurvature[i] = diffRange * diffRange;  // diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i]          = 0; // cloudLabel 元素初始化为 0
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i]; // 存储点的曲率值
            cloudSmoothness[i].ind   = i; // 激光点的一维索引
        }
	}

	void markOccludedPoints() {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 6; ++i) {
            // occluded points
            float depth1   = cloudInfo.pointRange[i];
            float depth2   = cloudInfo.pointRange[i + 1];
            // 两个激光点的列数的差值: 
            // 同一条扫描线上的两个激光点对应的值为1
            // 若两个点之间有异常点(距离过大或过小的点)，该值可能会因为剔除异常点而稍大
            // 若第一个激光点是前一个扫描线的结束，而第二个激光点是后一个扫描线的开始，则该值很大
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));
            if (columnDiff < 10) { // 判断是否位于同一根扫描线
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3) { // 第 i 个点深度比第 i+1 个点深度大超过 0.3 米，说明若激光雷达向第 i+1 个点方向移动，很可能因为被遮挡而看不到第 i 个点
                    // 标记遮挡点
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i]     = 1;
                    continue;
                } else if (depth2 - depth1 > 0.3) { // 第 i+1 个点深度比第 i 个点深度大超过 0.3 米，说明若激光雷达向第 i 个点方向移动，很可能因为被遮挡而看不到第 i+1 个点
                    // 标记遮挡点
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                    continue;
                }
            }
            float diff1 = std::abs(float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

            // 第 i-1 个点和第 i 个点的深度之差与第 i+1 个点和第 i 个点的深度之差均大于第 i 个点深度的 0.02 倍，则认为第 i 个点位于其与所属激光雷达光束平行的平面上
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1; // 标记平行点
        }
	}

	void extractFeatures() {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++) { // 遍历所有扫描线
            surfaceCloudScan->clear();

            // 将一条扫描线扫描一周的数据等分成 6 段，在每份中单独提取特征
            for (int j = 0; j < 6; j++) {
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6; // sp: start point
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1; // ep: end point

                if (sp >= ep)
                    continue;

                // 按照曲率从小到大排序 cloudSmoothness.begin() + sp 到 cloudSmoothness.begin() + ep
                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                int largestPickedNum = 0;
                // 按曲率从大到小遍历点云
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSmoothness[k].ind; // 点的一维索引
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold) { // 若该点非遮挡点和平行点，其曲率也大于边点阈值 edgeThreshold(0.1)
                        largestPickedNum++;
                        if (largestPickedNum <= 20) { // 每段扫描数据至多提取 20 个角点(边缘点)
                            cloudLabel[ind] = 1; // 标记为角点
                            cornerCloud->push_back(extractedCloud->points[ind]); // 添加到角点点云
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1; // 标记为已被处理
                        // 同一条扫描线上的后 5 个点标记不再处理
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        // 同一条扫描线上的前 5 个点标记不再处理
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 按曲率从小到大遍历点云
                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold) { // 若该点非遮挡点和平行点，其曲率也小于面点阈值 surfThreshold(0.1)
                        cloudLabel[ind]          = -1; // 标记为面点
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                        cloudNeighborPicked[ind] = 1; // 标记为已被处理

                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }

                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {
                    if (cloudLabel[k] <= 0) { 
                        surfaceCloudScan->push_back(extractedCloud->points[k]); // 平面点和未被处理的点，都认为是平面点，加入平面点云集合
                    }
                }
            }

            // 平面点云降采样
            surfaceCloudScanDS->clear(); // DS: down sampled
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS); // 对输入点云 surfaceCloudScan 进行降采样，降采样后的点云数据存储在 surfaceCloudScanDS 中

            *surfaceCloud += *surfaceCloudScanDS;
        }
	}

	void freeCloudInfoMemory() {
        // 清理cloudInfo 中不再需要的信息
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
	}

	void publishFeatureCloud() {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints, cornerCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
	}
};

int main(int argc, char **argv) {
	ros::init(argc, argv, "lio_track");

	FeatureExtraction FE;

	ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");

	ros::spin();

	return 0;
}