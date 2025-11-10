#include "solver.h"
#include <gtsam/nonlinear/ISAM2-impl.h>

gtsam::KeySet gatherMaxMixtureRelinearizationKeys(const gtsam::NonlinearFactorGraph nonlinearFactors, // 非线性因子
												  const gtsam::Values theta, // 优化变量的当前解
												  const gtsam::VectorValues delta, // 更新增量
												  gtsam::KeySet* markedKeys) { // 存储已经被标记为需要重新线性化的键
	gtsam::KeySet relinKeys; // 存储需要重新线性化的键
	LooselyCoupledDetectionFactor* lcdf;
	TightlyCoupledDetectionFactor* tcdf;

	for (const auto& factor : nonlinearFactors) { // 遍历因子
        // 通过 dynamic_cast 尝试将每个因子转换为 LooselyCoupledDetectionFactor 或 TightlyCoupledDetectionFactor 类型
		lcdf = dynamic_cast<LooselyCoupledDetectionFactor*>(factor.get());
		tcdf = dynamic_cast<TightlyCoupledDetectionFactor*>(factor.get());

		// 如果因子无法转换为 LooselyCoupledDetectionFactor 或 TightlyCoupledDetectionFactor，则跳过该因子
		if (lcdf == nullptr && tcdf == nullptr) continue;

		int index;
		int cachedIndex;
		double error;
		const std::vector<Detection>* detections;
		gtsam::Key robotPoseKey;
		gtsam::Key objectPoseKey;
		gtsam::Pose3 robotPose;
		gtsam::Pose3 objectPose;
        jsk_recognition_msgs::BoundingBox objectDetection;
        gtsam::Pose3 objectVelocity;

		// 根据转换后因子的类型(lcdf 或 tcdf), 提取相关的检测数据、机器人姿态键、物体姿态键以及缓存的检测索引
        if (lcdf != nullptr) {
			detections    = &lcdf->getDetections();
			robotPoseKey  = lcdf->robotPoseKey();
			objectPoseKey = lcdf->objectPoseKey();
			cachedIndex   = lcdf->getCachedDetectionIndex();
            objectDetection = lcdf->getObjectDetection();
            objectVelocity = lcdf->getObjectVelocity();
		} else {  // tcdf != nullptr
			detections    = &tcdf->getDetections();
			robotPoseKey  = tcdf->robotPoseKey();
			objectPoseKey = tcdf->objectPoseKey();
			cachedIndex   = tcdf->getCachedDetectionIndex();
            objectDetection = tcdf->getObjectDetection();
            objectVelocity = tcdf->getObjectVelocity();
		}

		robotPose  = gtsam::traits<gtsam::Pose3>::Retract(theta.at<gtsam::Pose3>(robotPoseKey),
														  delta.at(robotPoseKey)); // T = T_{w,l_{i-1}}*T_{l_{i-1},l_{i}} --> T_{w,l_{i}}
		objectPose = gtsam::traits<gtsam::Pose3>::Retract(theta.at<gtsam::Pose3>(objectPoseKey),
														  delta.at(objectPoseKey)); // T = T_{w,obj_{i}}*T_{obj_{i-1},obj_{i}} --> T_{w,obj_{i}}

		std::tie(index, error) = getDetectionIndexAndError(robotPose.inverse() * objectPose, objectDetection, objectVelocity, *detections); // 计算更新后的姿态相对于检测数据的最佳匹配索引 index 和相应的误差 error T_{w,l_{i}}^{-1}*T_{w,obj_{i}} --> T_{l_{i},obj_{i}}

		if (index != cachedIndex) { // 如果计算出的索引 index 与 cachedIndex 不同，则认为该因子的线性化点发生了变化，需要重新线性化
            // 将机器人姿态键、物体姿态键添加到 relinKeys 和 markedKeys 中
            cout << "relinearized" << endl;
			relinKeys.insert(robotPoseKey);
			relinKeys.insert(objectPoseKey);
			markedKeys->insert(robotPoseKey);
			markedKeys->insert(objectPoseKey);
		}
	}

	return relinKeys;
}

gtsam::ISAM2Result
MaxMixtureISAM2::update(
		const gtsam::NonlinearFactorGraph& newFactors,
		const gtsam::Values& newTheta,
		const gtsam::FactorIndices& removeFactorIndices, // 要移除的因子的索引集合
		const boost::optional<gtsam::FastMap<gtsam::Key, int> >& constrainedKeys, // 可选参数，表示受约束的关键字及其对应的约束类型
		const boost::optional<gtsam::FastList<gtsam::Key> >& noRelinKeys, // 可选参数，表示不需要重新线性化的关键字集合
		const boost::optional<gtsam::FastList<gtsam::Key> >& extraReelimKeys, // 可选参数，表示额外需要重新消元的关键字集合
		bool force_relinearize) { // 布尔标志，指示是否强制重新线性化所有变量
	gtsam::ISAM2UpdateParams params; // 这是一个结构体，用于存储更新 iSAM2 优化器时所需的参数
	params.constrainedKeys     = constrainedKeys;
	params.extraReelimKeys     = extraReelimKeys;
	params.force_relinearize   = force_relinearize;
	params.noRelinKeys         = noRelinKeys;
	params.removeFactorIndices = removeFactorIndices;

	return update(newFactors, newTheta, params);
}

gtsam::ISAM2Result
MaxMixtureISAM2::update(const gtsam::NonlinearFactorGraph& newFactors, // 非线性因子
						const gtsam::Values& newTheta, // 初始估计值 
						const gtsam::ISAM2UpdateParams& updateParams) { // 包含更新参数的结构体，如是否强制重新线性化、需要额外消元的键等
	
    gttic(ISAM2_update); // 记录时间戳，用于性能分析
	this->update_count_ += 1; // 更新计数器，记录这是第几次调用 update 函数
	gtsam::UpdateImpl::LogStartingUpdate(newFactors, *this); // 记录开始更新的日志信息
	gtsam::ISAM2Result result(params_.enableDetailedResults); // 创建一个 gtsam::ISAM2Result 对象，用于存储更新结果
	gtsam::UpdateImpl update(params_, updateParams); // 创建一个 gtsam::UpdateImpl 对象，用于执行具体的更新操作

	if (update.relinarizationNeeded(update_count_)) // 检查是否需要重新线性化
		updateDelta(updateParams.forceFullSolve); // 如果需要重新线性化，则更新增量 delta_

	update.pushBackFactors(newFactors, &nonlinearFactors_, &linearFactors_,
						   &variableIndex_, &result.newFactorsIndices,
						   &result.keysWithRemovedFactors); // 将新因子添加到现有的非线性因子图中
	update.computeUnusedKeys(newFactors, variableIndex_,
							 result.keysWithRemovedFactors, &result.unusedKeys); // 计算不再使用的键，并将其存储在 result.unusedKeys

	addVariables(newTheta, result.details()); // 将新的变量添加到当前的估计值中
	if (params_.evaluateNonlinearError) // 如果配置中启用了非线性误差评估
		update.error(nonlinearFactors_, calculateEstimate(), &result.errorBefore); // 计算更新前的误差，并存储在 result.errorBefore 中

	update.gatherInvolvedKeys(newFactors, nonlinearFactors_,
							  result.keysWithRemovedFactors, &result.markedKeys); // 收集所有涉及新因子或被移除因子的键，添加到 markedKeys 中
	update.updateKeys(result.markedKeys, &result); // 更新 markedKeys 中的关键变量的状态到 result 对象中

	gtsam::KeySet relinKeys;
	result.variablesRelinearized = 0;
	if (update.relinarizationNeeded(update_count_)) {
		relinKeys = update.gatherRelinearizeKeys(roots_, delta_, fixedVariables_, &result.markedKeys); // 收集增量 delta_ 超过某个阈值的键
		relinKeys.merge(gatherMaxMixtureRelinearizationKeys(nonlinearFactors_,
															theta_,
															delta_,
															&result.markedKeys)); // 收集最大混合模型发生变化的键，并通过 merge 合并到 relinKeys

		update.recordRelinearizeDetail(relinKeys, result.details());
		if (!relinKeys.empty()) {
			update.findFluid(roots_, relinKeys, &result.markedKeys, result.details()); // 标记涉及重新线性化变量及其祖先的所有团
			gtsam::UpdateImpl::ExpmapMasked(delta_, relinKeys, &theta_); // 更新重新线性化变量的线性化点，即将增量应用到当前估计值上
		}
		result.variablesRelinearized = result.markedKeys.size(); // 记录实际重新线性化的变量数量
	}

	update.linearizeNewFactors(newFactors, theta_, nonlinearFactors_.size(),
							   result.newFactorsIndices, &linearFactors_); // 将新因子线性化
	update.augmentVariableIndex(newFactors, result.newFactorsIndices,
								&variableIndex_); // 更新变量索引，以反映新因子和新变量的加入

	recalculate(updateParams, relinKeys, &result); // 重新计算贝叶斯树的顶部，并更新相关数据结构
	if (!result.unusedKeys.empty()) removeVariables(result.unusedKeys); // 移除不再使用的变量
	result.cliques = this->nodes().size(); // 记录当前贝叶斯树中的团数量

	if (params_.evaluateNonlinearError) // 如果配置中启用了非线性误差评估
		update.error(nonlinearFactors_, calculateEstimate(), &result.errorAfter); // 计算更新后的误差，并存储在 result.errorAfter 中
	return result;
}
