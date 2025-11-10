#include "factor.h"

#include <boost/format.hpp>

#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose3.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace gtsam;
/* -------------------------------------------------------------------------- */
/*                                  Detection                                 */
/* -------------------------------------------------------------------------- */

Detection::Detection(Detection::BoundingBox box,
					 gtsam::Vector6 variances)
		: box(box), variances(variances) {
	auto p     = box.pose.position;
	auto q     = box.pose.orientation;
	this->pose = gtsam::Pose3(gtsam::Rot3(q.w, q.x, q.y, q.z), gtsam::Point3(p.x, p.y, p.z));
}

/* -------------------------------------------------------------------------- */

const double Detection::errorDynamic(const gtsam::Pose3 x) const {
    // 使用 GTSAM 提供的 traits 提取局部误差向量
    gtsam::Vector errorVector = gtsam::traits<gtsam::Pose3>::Local(this->pose, x);
    // 将误差向量白化（乘以协方差的逆平方根）
    gtsam::Vector whitenedError = this->getDiagonal()->whiten(errorVector);
    // Mahalanobis 距离即为白化向量的 L2 范数
    return whitenedError.norm();
  }
  

const double Detection::errorStatic(const gtsam::Pose3 x, jsk_recognition_msgs::BoundingBox objectDetection) const { 
    Vec3T t1, t2;
    Mat3T cov1, cov2;
    cov1 = Mat3T::Zero();
    cov2 = Mat3T::Zero();
    auto pose1 = gtsam::Pose3(gtsam::Rot3::Quaternion(this->box.pose.orientation.w, this->box.pose.orientation.x, this->box.pose.orientation.y, this->box.pose.orientation.z),
                              gtsam::Point3(this->box.pose.position.x, this->box.pose.position.y, this->box.pose.position.z)); // estT_{l,obj}
    auto pose2 = x; // T_{l,obj}
    t1 = pose1.translation();
    t2 = pose2.translation();
    cov1.diagonal() << pow(this->box.dimensions.x / 6, 2), pow(this->box.dimensions.y / 6, 2), pow(this->box.dimensions.z / 6, 2);
    cov2.diagonal() << pow(objectDetection.dimensions.x / 6, 2), pow(objectDetection.dimensions.y / 6, 2), pow(objectDetection.dimensions.z / 6, 2);
    // auto R1 = pose1.rotation();
    // auto R2 = pose2.rotation();
    // cov1 = R1 * cov1 * R1.transpose(); // = cov1
    // cov2 = R2 * cov2 * R2.transpose(); // = cov2
    // gtsam::Rot3::Logmap(R1.inverse() * R2).norm()
    return 0.5 * (cov2.inverse() * cov1 + cov1.inverse() * cov2 - 2 * Mat3T::Identity()).trace() + 0.5 * (t1 -t2).transpose()*(cov1.inverse() + cov2.inverse())*(t1 -t2);
}

/* -------------------------------------------------------------------------- */

std::tuple<size_t, double>
getDynamicDetectionIndexAndError(const gtsam::Pose3 &d, std::vector<Detection> detections) { // 在给定的检测列表中找到与给定位姿 d 最匹配的检测，并返回该检测的索引和对应的误差
	if (detections.size() == 0) {
		throw std::runtime_error("Does not receive any detection.");
	}

	size_t idx   = 0;
	double error = detections[0].errorDynamic(d); // 给定位姿 d, 第一个检测的误差

	// Figure out the optimal detection index
	for (size_t i = 1; i < detections.size(); ++i) {
		double error_i = detections[i].errorDynamic(d);
		if (error_i < error) { // 如果当前检测的误差小于之前记录的最小误差，则更新最佳匹配的索引 idx 和对应的误差 error
			idx   = i;
			error = error_i;
		}
	}

	return std::make_tuple(idx, error);
}

std::tuple<size_t, double>
getStaticDetectionIndexAndError(const gtsam::Pose3 &d, jsk_recognition_msgs::BoundingBox objectDetection, std::vector<Detection> detections) { // 在给定的检测列表中找到与给定位姿 d 最匹配的检测，并返回该检测的索引和对应的误差
	if (detections.size() == 0) {
		throw std::runtime_error("Does not receive any detection.");
	}

	size_t idx   = 0;
	double error = detections[0].errorStatic(d, objectDetection); // 给定位姿 d, 第一个检测的误差

	// Figure out the optimal detection index
	for (size_t i = 1; i < detections.size(); ++i) {
		double error_i = detections[i].errorStatic(d, objectDetection);
		if (error_i < error) { // 如果当前检测的误差小于之前记录的最小误差，则更新最佳匹配的索引 idx 和对应的误差 error
			idx   = i;
			error = error_i;
		}
	}
	return std::make_tuple(idx, error);
}

std::tuple<size_t, double>
getDetectionIndexAndError(const gtsam::Pose3 &d, jsk_recognition_msgs::BoundingBox objectDetection, gtsam::Pose3 objectVelocity, std::vector<Detection> detections) {
    double volume = objectDetection.dimensions.x * objectDetection.dimensions.y * objectDetection.dimensions.z;
    auto identity = Pose3::identity();
    double linearVelocityScale = gtsam::traits<Pose3>::Local(identity, objectVelocity).head<3>().norm();
    if(volume < 0.1 || linearVelocityScale > 1) {
        return getDynamicDetectionIndexAndError(d, detections);
    } else {
        return getStaticDetectionIndexAndError(d, objectDetection, detections);
    }
}

/* -------------------------------------------------------------------------- */
/*                      Tightly-Coupled Detection Factor                      */
/* -------------------------------------------------------------------------- */

gtsam::Vector TightlyCoupledDetectionFactor::evaluateError(const gtsam::Pose3 &robotPose,
											 const gtsam::Pose3 &objectPose,
											 const gtsam::Pose3 &measured,
											 boost::optional<gtsam::Matrix &> H1,
											 boost::optional<gtsam::Matrix &> H2) const {
	// template<class Class> static Class Between(const Class& m1, const Class& m2, ChartJacobian H1 = boost::none, ChartJacobian H2 = boost::none)
    // m1: 第一个几何对象，通常表示自车或参考坐标系的位姿或位置
    // m2: 第二个几何对象，通常表示物体或目标坐标系的位姿或位置
    // H1 和 H2: 这两个参数是可选的雅可比矩阵引用，分别表示残差对 m1 和 m2 的偏导数
    // 对于 Point3 类型：Between 会计算 m1 和 m2 之间的向量差，即 m1 - m2; 对于 Pose3 类型：函数返回的是一个 Pose3 对象 m1 * m2.inverse(); 
    // 如果提供了 H1 和 H2 参数，Between 函数还会计算并返回残差对 m1 和 m2 的雅可比矩阵
    gtsam::Pose3 hx = gtsam::traits<gtsam::Pose3>::Between(robotPose,
														   objectPose,
													       H1,
													       H2); // T_{w,l}^{-1}*T_{w,obj} --> T_{l,obj}
	return gtsam::traits<gtsam::Pose3>::Local(measured, hx); // (estT_{l,obj})^{-1}*T_{l,obj} = exp(ξ)
}

/* -------------------------------------------------------------------------- */

// 计算因子的未白化(unwhitened)残差，评估当前估计值与测量值之间的差异
gtsam::Vector TightlyCoupledDetectionFactor::unwhitenedError(const gtsam::Pose3 &measured, // 测量值
											                 const gtsam::Values &x, // 当前优化问题中所有变量的估计值
											                 boost::optional<std::vector<gtsam::Matrix> &> H) const { // 可选参数，表示雅可比矩阵的引用。如果提供了该参数，unwhitenedError 方法会计算并返回残差以及相应的雅可比矩阵
	if (this->active(x)) { // 用于检查因子是否在当前优化迭代中是活跃的。因子是否活跃取决于其涉及的变量是否在当前的优化问题中。
		const gtsam::Pose3 &x1 = x.at<gtsam::Pose3>(robotPoseKey()); // 自车的位姿
		const gtsam::Pose3 &x2 = x.at<gtsam::Pose3>(objectPoseKey()); // 物体的位姿
		if (H) {
			return evaluateError(x1, x2, measured, (*H)[0], (*H)[1]);
		} else {
			return evaluateError(x1, x2, measured);
		}
	} else {
		return gtsam::Vector::Zero(this->dim());
	}
}

/* -------------------------------------------------------------------------- */

boost::shared_ptr<gtsam::GaussianFactor>
TightlyCoupledDetectionFactor::linearize(const gtsam::Values &c) const { // c: 因子
	
	if (!active(c))
		return boost::shared_ptr<gtsam::JacobianFactor>();

	double error;
	const auto robotPose                  = c.at<gtsam::Pose3>(this->robotPoseKey());
	const auto objectPose                 = c.at<gtsam::Pose3>(this->objectPoseKey());
	std::tie(cachedDetectionIndex, error) = getDetectionIndexAndError(robotPose.inverse() * objectPose, this->objectDetection, this->objectVelocity, this->detections);

	auto measured   = this->detections[cachedDetectionIndex].getPose();
	auto noiseModel = this->noiseModels[cachedDetectionIndex]; // Σ

	std::vector<gtsam::Matrix> A(size());
	gtsam::Vector b = -unwhitenedError(measured, c, A); // 计算雅可比矩阵存入 A, 计算因子的未加权残差的负值存入 b
	if (noiseModel && b.size() != noiseModel->dim()) // 检查残差的维度是否与噪声模型的维度一致
		throw std::invalid_argument(
				boost::str(
						boost::format(
								"NoiseModelFactor: NoiseModel has dimension %1% instead of %2%.") % noiseModel->dim() % b.size()));

	if (noiseModel)
		noiseModel->WhitenSystem(A, b); // 白化残差和雅可比矩阵: Σ^{-1/2}*A, Σ^{-1/2}*b

	std::vector<std::pair<gtsam::Key, gtsam::Matrix> > terms(size()); // 用于存储变量键和雅可比矩阵。terms[j].first 是变量的键，terms[j].second 是对应的雅可比矩阵
	for (size_t j = 0; j < size(); ++j) {
		terms[j].first = keys()[j]; // 第 j 个变量的键
		terms[j].second.swap(A[j]); // 将 A[j] 中的值移动到 terms[j].second 中，并清空 A[j]
	}

	// TODO pass unwhitened + noise model to Gaussian factor
	using gtsam::noiseModel::Constrained;
	if (noiseModel && noiseModel->isConstrained()) // false
		return gtsam::GaussianFactor::shared_ptr( 
			new gtsam::JacobianFactor(terms, b, boost::static_pointer_cast<Constrained>(noiseModel)->unit())); // JacobianFactor 是 GTSAM 中的一种线性因子，表示线性化的约束。它由雅可比矩阵、残差向量 b 和噪声模型组成
	else
		return gtsam::GaussianFactor::shared_ptr(new gtsam::JacobianFactor(terms, b));
}

/* -------------------------------------------------------------------------- */

double TightlyCoupledDetectionFactor::error(const gtsam::Values &c) const {
	if (active(c)) {
		// Determine which detection is used to generate the factor function, a.k.a.
		// the Max-Mixture model
		size_t index;
		double error;
		const auto robotPose   = c.at<gtsam::Pose3>(this->robotPoseKey());
		const auto objectPose  = c.at<gtsam::Pose3>(this->objectPoseKey());
		std::tie(index, error) = getDetectionIndexAndError(robotPose.inverse() * objectPose, this->objectDetection, this->objectVelocity, this->detections);

		auto measured   = this->detections[index].getPose();
		auto noiseModel = this->noiseModels[index];

		const gtsam::Vector b = unwhitenedError(measured, c);
		if (noiseModel && b.size() != noiseModel->dim())
			throw std::invalid_argument(
					boost::str(
						boost::format(
							"NoiseModelFactor: NoiseModel has dimension %1% instead of %2%.") % noiseModel->dim() % b.size()));
		return 0.5 * 0.5 * noiseModel->whiten(b).squaredNorm();
	} else {
		return 0.0;
	}
}

/* -------------------------------------------------------------------------- */
/*                      Loosely-Coupled Detection Factor                      */
/* -------------------------------------------------------------------------- */

gtsam::Vector
LooselyCoupledDetectionFactor::evaluateError(const gtsam::Pose3 &robotPose,
											 const gtsam::Pose3 &objectPose,
											 const gtsam::Pose3 &measured,
											 boost::optional<gtsam::Matrix &> H1) const {
	gtsam::Pose3 hx = gtsam::traits<gtsam::Pose3>::Between(robotPose,
														   objectPose,
														   boost::none,
														   H1);
	return gtsam::traits<gtsam::Pose3>::Local(measured, hx);
}

/* -------------------------------------------------------------------------- */

gtsam::Vector
LooselyCoupledDetectionFactor::unwhitenedError(const gtsam::Pose3 &measured,
											   const gtsam::Values &x,
											   boost::optional<std::vector<gtsam::Matrix> &> H) const {
	if (this->active(x)) {
		const gtsam::Pose3 &x1 = x.at<gtsam::Pose3>(this->robotPoseKey());
		const gtsam::Pose3 &x2 = x.at<gtsam::Pose3>(this->objectPoseKey());
		if (H) {
			return evaluateError(x1, x2, measured, (*H)[0]);
		} else {
			return evaluateError(x1, x2, measured);
		}
	} else {
		return gtsam::Vector::Zero(this->dim());
	}
}

/* -------------------------------------------------------------------------- */

boost::shared_ptr<gtsam::GaussianFactor>
LooselyCoupledDetectionFactor::linearize(const gtsam::Values &c) const {
	// Only linearize if the factor is active
	if (!active(c))
		return boost::shared_ptr<gtsam::JacobianFactor>();

	double error;
	const auto robotPose                  = c.at<gtsam::Pose3>(this->robotPoseKey());
	const auto objectPose                 = c.at<gtsam::Pose3>(this->objectPoseKey());
	std::tie(cachedDetectionIndex, error) = getDetectionIndexAndError(robotPose.inverse() * objectPose, this->objectDetection, this->objectVelocity, this->detections);

	auto measured   = this->detections[cachedDetectionIndex].getPose();
	auto noiseModel = this->noiseModels[cachedDetectionIndex];

	// Call evaluate error to get Jacobians and RHS(Right-Hand Side) vector b
	std::vector<gtsam::Matrix> A(size());
	gtsam::Vector b = -unwhitenedError(measured, c, A);
	if (noiseModel && b.size() != noiseModel->dim())
		throw std::invalid_argument(
				boost::str(
					boost::format("NoiseModelFactor: NoiseModel has dimension %1% instead of %2%.") % noiseModel->dim() % b.size()));

	// Whiten the corresponding system now
	if (noiseModel)
		noiseModel->WhitenSystem(A, b);

	// Fill in terms, needed to create JacobianFactor below
	std::vector<std::pair<gtsam::Key, gtsam::Matrix> > terms(size());
	for (size_t j = 0; j < size(); ++j) {
		terms[j].first = keys()[j];
		terms[j].second.swap(A[j]);
	}

	// TODO pass unwhitened + noise model to Gaussian factor
	using gtsam::noiseModel::Constrained;
	if (noiseModel && noiseModel->isConstrained())
		return gtsam::GaussianFactor::shared_ptr(
			new gtsam::JacobianFactor(terms, b, boost::static_pointer_cast<Constrained>(noiseModel)->unit()));
	else
		return gtsam::GaussianFactor::shared_ptr(new gtsam::JacobianFactor(terms, b));
}

/* -------------------------------------------------------------------------- */

double LooselyCoupledDetectionFactor::error(const gtsam::Values &c) const {
	if (active(c)) {
		// Determine which detection is used to generate the factor function, a.k.a.
		// the Max-Mixture model
		size_t index;
		double error;
		const auto robotPose   = c.at<gtsam::Pose3>(this->robotPoseKey());
		const auto objectPose  = c.at<gtsam::Pose3>(this->objectPoseKey());
		std::tie(index, error) = getDetectionIndexAndError(robotPose.inverse() * objectPose, this->objectDetection, this->objectVelocity, this->detections);

		auto measured   = this->detections[index].getPose();
		auto noiseModel = this->noiseModels[index];

		const gtsam::Vector b = unwhitenedError(measured, c);
		if (noiseModel && b.size() != noiseModel->dim())
			throw std::invalid_argument(
					boost::str(
						boost::format(
							"NoiseModelFactor: NoiseModel has dimension %1% instead of %2%.") % noiseModel->dim() % b.size()));
		return 0.5 * 0.5 * noiseModel->whiten(b).squaredNorm();
	} else {
		return 0.0;
	}
}

/* -------------------------------------------------------------------------- */
/*                             Stable Pose Factor                             */
/* -------------------------------------------------------------------------- */

gtsam::Vector
StablePoseFactor::evaluateError(const gtsam::Pose3 &previousPose,
								const gtsam::Pose3 &velocity,
								const gtsam::Pose3 &nextPose,
								boost::optional<gtsam::Matrix &> H1,
								boost::optional<gtsam::Matrix &> H2,
								boost::optional<gtsam::Matrix &> H3) const {
	auto identity = gtsam::Pose3::identity();

	auto deltaPoseVec = gtsam::traits<gtsam::Pose3>::Local(identity, velocity) * this->deltaTime; // Log(v) * Δt
	auto deltaPose    = gtsam::traits<gtsam::Pose3>::Retract(identity, deltaPoseVec); // estT_{i-1,i} = exp(Log(v) * Δt)
	gtsam::Pose3 hx   = nextPose.inverse() * previousPose * deltaPose; // T_{w,i}^{-1}*T_{w,i-1}*estT_{i-1,i} --> T_{w,i}^{-1}*estT_{w,i} 表示物体的实际位姿变化与预测的位姿变化之间的误差

	if (H1) *H1 = deltaPose.inverse().AdjointMap();
	if (H2) *H2 = this->deltaTime * gtsam::Pose3::ExpmapDerivative(deltaPoseVec) * gtsam::Pose3::LogmapDerivative(velocity);
	if (H3) *H3 = -hx.inverse().AdjointMap();

	return gtsam::traits<gtsam::Pose3>::Local(identity, hx);
}
