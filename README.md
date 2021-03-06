# UnscentedKalmanFilterCA
Unscented Kalman Filter with [uniform acceleration](https://en.wikipedia.org/wiki/Acceleration#Uniform_acceleration) kinematic model.

Used UKF from the [opencv_contrib](https://github.com/opencv/opencv_contrib) tracking library.

Demo of filtering the measurement of the centroid and rectangle with hardcore noise:
- **Blue** - cursor with dispersion noise.
- **Green** - UKF prediction.
- **Red** - UKF correction.

<div align="center">
  <img src="other/cursor_demo.png", width="512">
</div>
