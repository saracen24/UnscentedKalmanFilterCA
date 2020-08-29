#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "motion_model.hpp"
#include "sliding_vector.hpp"

using namespace std;
using namespace cv;

static RNG sRng(0xFFFFFFFF);
static const float kCDispersion = 64.f;
static const float kSDispersion = 64.f;
static const Size2f kRectSize(64.f, 128.f);

static void MouseCallbackFunc(int event, int x, int y, int flags,
                              void* userdata) {
  if (event == EVENT_MOUSEMOVE) {
    aiz::RigidBody* const body = static_cast<aiz::RigidBody* const>(userdata);
    sRng.next();
    body->centroid.x = x + sRng.uniform(-kCDispersion, kCDispersion);
    body->centroid.y = y + sRng.uniform(-kCDispersion, kCDispersion);
    body->size.width =
        kRectSize.width + sRng.uniform(-kSDispersion, kSDispersion);
    body->size.height =
        kRectSize.height + sRng.uniform(-kSDispersion, kSDispersion);
  }
}

int main() {
  Size wsize(1024, 1024);
  Mat canvas = Mat::zeros(wsize, CV_8UC3);

  aiz::RigidBody body{Point(canvas.cols * 0.5f, canvas.rows * 0.5f), kRectSize};

  const String windowName("Canvas");
  namedWindow(windowName, WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL);
  resizeWindow(windowName, Size(canvas.size()));

  setMouseCallback(windowName, MouseCallbackFunc, &body);

  aiz::MotionFilterSettings mfcfg;
  mfcfg.centroidMNCov = 1e-1f;
  aiz::MotionModel mm(body, mfcfg);

  const size_t kHistory = 512;
  aiz::SlidingVector<Point2f> cursorSv(kHistory), predictedSv(kHistory),
      correctedSv(kHistory);

  while (true) {
    canvas = Mat::zeros(wsize, CV_8UC3);

    aiz::RigidBody predicted = mm.predict();
    aiz::RigidBody corrected = mm.correct(body);

    cursorSv.pushBack(body.centroid);
    predictedSv.pushBack(predicted.centroid);
    correctedSv.pushBack(corrected.centroid);

    drawMarker(canvas, body.centroid, CV_RGB(0, 0, 255), MARKER_TILTED_CROSS,
               16, 5, LINE_8);
    drawMarker(canvas, predicted.centroid, CV_RGB(0, 255, 0),
               MARKER_TILTED_CROSS, 8, 3, LINE_8);
    drawMarker(canvas, corrected.centroid, CV_RGB(255, 0, 0),
               MARKER_TILTED_CROSS, 4, 1, LINE_8);

    Rect2f cursorRect(body.centroid.x - body.size.width * 0.5f,
                      body.centroid.y - body.size.height * 0.5f,
                      body.size.width, body.size.height);
    rectangle(canvas, cursorRect, CV_RGB(0, 0, 255), 1, LINE_8);

    Rect2f correctedRect(corrected.centroid.x - corrected.size.width * 0.5f,
                         corrected.centroid.y - corrected.size.height * 0.5f,
                         corrected.size.width, corrected.size.height);
    rectangle(canvas, correctedRect, CV_RGB(255, 0, 0), 1, LINE_8);

    const auto& cursorv = cursorSv.vecCRef();
    const auto& predictedv = predictedSv.vecCRef();
    const auto& correctedv = correctedSv.vecCRef();

    if (!cursorv.empty())
      for (size_t i = 0; i < cursorv.size() - 1; i++)
        line(canvas, cursorv[i], cursorv[i + 1], CV_RGB(0, 0, 255), 2, LINE_AA);

    if (!predictedv.empty())
      for (size_t i = 0; i < predictedv.size() - 1; i++)
        line(canvas, predictedv[i], predictedv[i + 1], CV_RGB(0, 255, 0), 1,
             LINE_AA);

    if (!correctedv.empty())
      for (size_t i = 0; i < correctedv.size() - 1; i++)
        line(canvas, correctedv[i], correctedv[i + 1], CV_RGB(255, 0, 0), 2,
             LINE_AA);

    imshow(windowName, canvas);
    const char key = static_cast<const char>(waitKeyEx(33) & 0xFF);
    if (key == 'q') break;
  }

  return 0;
}
