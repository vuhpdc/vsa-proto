
#ifndef __BASIC_CONTROLLER_H__
#define __BASIC_CONTROLLER_H__

#include "common.h"
#include "controller.h"
#include "filters.h"

class Controller::Impl {
public:
  Impl(unsigned int currModel) : currModel_m(currModel) {
    std::cout << "Basic Controller" << std::endl;
    // auto filter = new EWMAFilter();
    auto filter = new ErrorBasedFilter();
    filter_m.reset(filter);
  }
  int update(std::shared_ptr<DetectionObject>);
  int predict(std::shared_ptr<FrameObject>);

  /* This variable sets how many frame we can send before receiving any feeback.
   * When taking a decision the feedback information should not too stale, e.g.,
   * 1sec or 2sec. Therefore, we can set this variable as, FRAME_RATE * (number
   * of sec)
   */
  const unsigned int stalenessGap = 30 * 1;

private:
  unsigned int getAccModel(unsigned int largeModel) {
    double highestAcc = 0.0;
    unsigned int highestModel;
    // linear search with O(n) complexity. This should be fine, because there
    // won't be very large number of models. But the search can be improved.
    for (auto i = 0; i <= largeModel; ++i) {
      double acc = ConfigManager::Default()->modelsAccuracy[i];
      if (highestAcc <= acc) {
        highestAcc = acc;
        highestModel = i;
      }
    }
    return highestModel;
  }

  void reset() {
    controlBuffer_m.clear();
    pos_m = 0;
    filter_m->reset(); // TODO: Check, if we can live without resetting.
  }

  vector<double> controlBuffer_m;
  int pos_m = 0;
  int controlWindow_m = CONTROL_WINDOW;
  double score_m = 0.0;
  std::atomic<unsigned int> currModel_m;
  std::unique_ptr<SmoothingFilter> filter_m;
};

int Controller::update(std::shared_ptr<DetectionObject> &detectionObject) {
  if (detectionObject->detectionHdr.frameHdr.frameId != iter_m) {
    return -1;
  }
  auto ret = _impl->update(detectionObject);
  ++iter_m; // TODO: needs to be threadsafe.
  return ret;
}

int Controller::predict(std::shared_ptr<FrameObject> &frameObject) const {
  // if (frameObject->frameHdr.frameId != iter_m) {
  while ((frameObject->frameHdr.frameId - iter_m) > _impl->stalenessGap) {
    // TODO: We wait for earlier frame to finish, so that feedback can be
    // incorporated for this frame.
    // return -1;
    // std::cout << frameObject->frameHdr.frameId << " predict sleep " <<
    // std::endl;
    usleep(500);
  }
  // std::cout << frameObject->frameHdr.frameId << " predict done " <<
  // std::endl;
  return _impl->predict(frameObject);
}

int Controller::Impl::update(std::shared_ptr<DetectionObject> detectionObject) {

  auto &detectionHdr = detectionObject->detectionHdr;
  auto &frameHdr = detectionObject->detectionHdr.frameHdr;
  // std::cout << "update: " << detectionHdr.usedModel << " : " << currModel_m
  // << std::endl;
  if (detectionHdr.usedModel != currModel_m) { // used model is not updated, so
                                               // server is still updating model
    return 0;
  }

  double time_spent = TIME_US_TO_MS(frameHdr.clientRecvTime -
                                    // frameHdr.captureTime);
                                    frameHdr.clientSendTime);

  filter_m->update(time_spent);
  auto smooth_time_spent = filter_m->predict();
  std::cout << "LatencySmooth:" << time_spent << ":" << smooth_time_spent << std::endl;
  time_spent = smooth_time_spent;

  bool on_time = (time_spent <= FRAME_DEADLINE) ? true : false;
  double diff = time_spent - FRAME_DEADLINE;
  diff = on_time ? -1 * pow(abs(diff), ON_TIME_EXP) : pow(diff, LATE_EXP);

  if (controlBuffer_m.size() ==
      controlWindow_m) { // control window full, start checking
    controlBuffer_m[pos_m] = diff;
    pos_m = (pos_m + 1) % controlWindow_m;
  } else { // control window not full yet, wait till its full
    controlBuffer_m.push_back(diff);
  }

  // TODO: We accumulate deviation error in the control buffer. However, after
  // we reset the control buffer, we should wait for control buffer to get
  // filled before taking any decision.
  double sum =
      std::accumulate(controlBuffer_m.begin(), controlBuffer_m.end(), 0.0);

  score_m = (score_m * HISTORY_WEIGHT) + ((1.0 - HISTORY_WEIGHT) * sum);

  // std::cout << frameHdr.frameId << " : " << time_spent << " : " << on_time <<
  // " : " << diff << " : "
  // << controlBuffer_m.size() << " : " << sum << std::endl;

  unsigned int prev_model = currModel_m;
  if (score_m <= -1 * UP_SUM * CONTROL_WINDOW) {

    if (currModel_m < MAX_MODEL) {
      currModel_m = getAccModel(currModel_m + 1);
    }
    // std::cout << frameHdr.frameId << " : update up to : " << currModel_m <<
    // std::endl;
  } else if (score_m >= DOWN_SUM * CONTROL_WINDOW) {
    if (currModel_m > MIN_MODEL) {
      currModel_m = getAccModel(currModel_m - 1);
    }
    // std::cout << frameHdr.frameId << " : update down to : " << currModel_m
    // << std::endl;
  }

  if (currModel_m != prev_model) { // reset the control buffer
    reset();
  }

  return 0;
}

int Controller::Impl::predict(std::shared_ptr<FrameObject> frameObject) {
  return currModel_m;
}

#endif /* __BASIC_CONTROLLER_H__ */
