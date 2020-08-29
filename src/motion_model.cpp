#include "motion_model.hpp"

#include <cmath>

using namespace std;
using namespace cv;

namespace aiz {

UniformAccelerationModel::UniformAccelerationModel(float deltaTime,
                                                   float accelerationFactor)
    : tracking::UkfSystemModel(),
      kDeltaTime(deltaTime),
      kAccelerationFactor(accelerationFactor) {}

void UniformAccelerationModel::stateConversionFunction(const Mat &x_k,
                                                       const Mat &u_k,
                                                       const Mat &v_k,
                                                       Mat &x_kplus1) {
  float x0 = x_k.at<float>(0, 0);
  float y0 = x_k.at<float>(1, 0);
  float w0 = x_k.at<float>(2, 0);
  float h0 = x_k.at<float>(3, 0);
  float vx0 = x_k.at<float>(4, 0);
  float vy0 = x_k.at<float>(5, 0);
  float a0 = x_k.at<float>(6, 0) * kAccelerationFactor;

  x_kplus1.at<float>(0, 0) =
      x0 + vx0 * kDeltaTime + a0 * pow(kDeltaTime, 2.f) * 0.5f;
  x_kplus1.at<float>(1, 0) =
      y0 + vy0 * kDeltaTime + a0 * pow(kDeltaTime, 2.f) * 0.5f;
  x_kplus1.at<float>(2, 0) = w0;
  x_kplus1.at<float>(3, 0) = h0;
  x_kplus1.at<float>(4, 0) = vx0 + a0 * kDeltaTime;
  x_kplus1.at<float>(5, 0) = vy0 + a0 * kDeltaTime;
  x_kplus1.at<float>(6, 0) = a0;

  if (u_k.size() == v_k.size())
    x_kplus1 += u_k + v_k;
  else
    x_kplus1 += v_k;
}

void UniformAccelerationModel::measurementFunction(const Mat &x_k,
                                                   const Mat &n_k, Mat &z_k) {
  float x0 = x_k.at<float>(0, 0);
  float y0 = x_k.at<float>(1, 0);
  float w0 = x_k.at<float>(2, 0);
  float h0 = x_k.at<float>(3, 0);
  float vx0 = x_k.at<float>(4, 0);
  float vy0 = x_k.at<float>(5, 0);
  float a0 = x_k.at<float>(6, 0) * kAccelerationFactor;

  z_k.at<float>(0, 0) = x0 + vx0 * kDeltaTime +
                        a0 * pow(kDeltaTime, 2.f) * 0.5f + n_k.at<float>(0, 0);
  z_k.at<float>(1, 0) = y0 + vy0 * kDeltaTime +
                        a0 * pow(kDeltaTime, 2.f) * 0.5f + n_k.at<float>(1, 0);
  z_k.at<float>(2, 0) = w0 + n_k.at<float>(2, 0);
  z_k.at<float>(3, 0) = h0 + n_k.at<float>(3, 0);
}

MotionModel::MotionModel(const RigidBody &body,
                         const MotionFilterSettings &cfg) {
  const int kDp = 7;
  const int kMp = 4;
  const int kCp = 0;
  const int kDataType = CV_32F;

  Ptr<UniformAccelerationModel> model =
      makePtr<UniformAccelerationModel>(cfg.deltaTime, cfg.accelerationFactor);
  tracking::UnscentedKalmanFilterParams params(kDp, kMp, kCp, 0., 0., model,
                                               kDataType);

  params.stateInit = Mat::zeros(kDp, 1, kDataType);
  params.stateInit.at<float>(0, 0) = body.centroid.x;
  params.stateInit.at<float>(1, 0) = body.centroid.y;
  params.stateInit.at<float>(2, 0) = body.size.width;
  params.stateInit.at<float>(3, 0) = body.size.height;

  params.errorCovInit = Mat::eye(kDp, kDp, kDataType) * cfg.errorCovInitFactor;

  params.processNoiseCov = Mat::zeros(kDp, kDp, kDataType);
  params.processNoiseCov.at<float>(0, 0) = cfg.centroidPNCov;
  params.processNoiseCov.at<float>(1, 1) = cfg.centroidPNCov;
  params.processNoiseCov.at<float>(2, 2) = cfg.sizePNCov;
  params.processNoiseCov.at<float>(3, 3) = cfg.sizePNCov;
  params.processNoiseCov.at<float>(4, 4) = cfg.velocityPNCov;
  params.processNoiseCov.at<float>(5, 5) = cfg.velocityPNCov;
  params.processNoiseCov.at<float>(6, 6) = cfg.accelerationPNCov;

  params.measurementNoiseCov = Mat::zeros(kMp, kMp, kDataType);
  params.measurementNoiseCov.at<float>(0, 0) = cfg.centroidMNCov;
  params.measurementNoiseCov.at<float>(1, 1) = cfg.centroidMNCov;
  params.measurementNoiseCov.at<float>(2, 2) = cfg.sizeMNCov;
  params.measurementNoiseCov.at<float>(3, 3) = cfg.sizeMNCov;

  params.alpha = cfg.alpha;
  params.k = cfg.kappa;
  params.beta = cfg.beta;

  mMeasurement = Mat(kMp, 1, kDataType);
  mUkf = tracking::createUnscentedKalmanFilter(params);
}

RigidBody MotionModel::predict() const {
  Mat prediction = mUkf->predict();
  return {Point2f(prediction.at<float>(0, 0), prediction.at<float>(1, 0)),
          Size2f(prediction.at<float>(2, 0), prediction.at<float>(3, 0))};
}

RigidBody MotionModel::correct(const RigidBody &body) const {
  mMeasurement.at<float>(0, 0) = body.centroid.x;
  mMeasurement.at<float>(1, 0) = body.centroid.y;
  mMeasurement.at<float>(2, 0) = body.size.width;
  mMeasurement.at<float>(3, 0) = body.size.height;

  Mat estimated = mUkf->correct(mMeasurement);
  return {Point2f(estimated.at<float>(0, 0), estimated.at<float>(1, 0)),
          Size2f(estimated.at<float>(2, 0), estimated.at<float>(3, 0))};
}

}  // namespace aiz
