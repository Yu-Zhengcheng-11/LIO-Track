# LIO-Track 随笔

## 1 SLAM部分

### **1.1 imageProjection**

#### 1.1.1 订阅

`subIMU`订阅话题`'imu_raw'`，该话题的消息发布自ROS Bag，表示原始IMU数据，当接受到该话题的IMU消息时，用回调函数`IMUHandler`进行处理

`subOdom`订阅话题`'odometry/imu_incremental'`，该话题的消息发布来自节点`/lio_track_imuPreintegration`，表示通过IMU预积分计算出的里程计位姿，当接受到该话题的消息时，用回调函数`odometryHandler`进行处理

`subLaserCloud`订阅话题`'points_raw'`，该话题的消息发布自ROS Bag，表示原始点云数据，点云中的每个点包含x,y,z,intensity,ring和时间戳等信息，当接受到该话题的点云消息时，用回调函数`cloudHandler`进行处理

#### 1.1.2 回调函数

`IMUHandler`：将新来的IMU消息加入到队列`odomQueue`中

`odometryHandler`：将新来的IMU预积分计算出的里程计位姿消息加入到队列`imuQueue`中

`cloudHandler`：

将新来的点云消息加入`cloudQueue`中，再从`cloudQueue`中取出最早的一帧作为当前帧，并将当前帧中的x,y,z,intensity信息转换为点云存入`rawCloudIn`；将当前帧中的全部信息(时间戳stamp，点云所在的坐标系frame_id，当前帧点云数width，是否包含NaN点的信息is_dense，单个元素为x,y,z,intensity,ring的数组fields)的转换为点云存入`laserCloudIn`；将当前帧中包含frame_id和stamp的消息头存入`cloudHeader`；将当前帧开始时间和结束时间分别存入`timeScanCur`和`timeScanEnd`

