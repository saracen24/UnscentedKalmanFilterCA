/**
 * @file motion_model.hpp.
 * @brief UKF with uniform acceleration motion model.
 * @author Aiz (c).
 * @date 2020.
 */
#pragma once

#include <opencv2/tracking/kalman_filters.hpp>

namespace aiz {

struct RigidBody {
  cv::Point2f centroid;
  cv::Size2f size;
};

struct MotionFilterSettings {
  float deltaTime = 1.f;
  float accelerationFactor = 1.f;
  float errorCovInitFactor = 1e-6f;
  float centroidPNCov = 1e-6f;
  float sizePNCov = 1e-6f;
  float velocityPNCov = 5e-6f;
  float accelerationPNCov = 1e-6f;
  float centroidMNCov = 1e-3f;
  float sizeMNCov = 1e-2f;
  float alpha = 1.f;
  float kappa = 0.f;
  float beta = 2.f;
};

class UniformAccelerationModel : public cv::tracking::UkfSystemModel {
 public:
  explicit UniformAccelerationModel(float deltaTime = 1.f,
                                    float accelerationFactor = 1.f);
  ~UniformAccelerationModel() override = default;

  void stateConversionFunction(const cv::Mat& x_k, const cv::Mat& u_k,
                               const cv::Mat& v_k, cv::Mat& x_kplus1) override;
  void measurementFunction(const cv::Mat& x_k, const cv::Mat& n_k,
                           cv::Mat& z_k) override;

 private:
  const float kDeltaTime;
  const float kAccelerationFactor;
};

class MotionModel {
 public:
  explicit MotionModel(const RigidBody& body, const MotionFilterSettings& cfg =
                                                  MotionFilterSettings());
  ~MotionModel() = default;

  RigidBody predict() const;
  RigidBody correct(const RigidBody& body) const;

 private:
  mutable cv::Mat mMeasurement;
  cv::Ptr<cv::tracking::UnscentedKalmanFilter> mUkf;
};

}  // namespace aiz
