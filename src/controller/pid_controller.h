
#ifndef __PID_CONTROLLER_H__
#define __PID_CONTROLLER_H__

#include "common.h"
#include "controller.h"
#include "filters.h"

class Controller::Impl {
public:
  Impl(unsigned int currModel) : currModel_m(currModel) {
    std::cout << "PID Controller" << std::endl;
    // auto filter = new EWMAFilter();
    auto filter = new ErrorBasedFilter();
    filter_m.reset(filter);

    prevModel_m = std::numeric_limits<int>::max();

    // adaptive threshold parameters
    errIncreasePerSize_m = UP_SUM;
    errVariance_m = DOWN_SUM;
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

  void getThresholds(double &low, double &high) {
    bool adaptive = false;
    if (!adaptive) {
      low = UP_SUM;
      high = DOWN_SUM;
    } else {
      low = errIncreasePerSize_m + errVariance_m;
      high = errVariance_m;
    }
  }

  void updateErrIncreasePerSize(unsigned int model, double timeSpent) {
    double alpha = 0.5;
    if (prevModel_m != currModel_m) {
      if (prevTimeSpent_m != -1.0) {
        double increase = abs(timeSpent - prevTimeSpent_m);
        errIncreasePerSize_m =
            alpha * errIncreasePerSize_m + (1 - alpha) * increase;
        // std::cout << "IncreasePerSize: " << increase << " " << errIncreasePerSize_m << std::endl;
      }
      prevTimeSpent_m = timeSpent;
    }
  }

  double pid_filter(double p_term, double i_term, double d_term) {
    double Ki = (1 - Kp) / time_m;
    double result = Kp * p_term;
    result += Ki * i_term;
    result += Kd * d_term;
    std::cout << "PID: " << p_term << " " << i_term << " " << d_term
              << std::endl;
    std::cout << "PID:Result: " << result << std::endl;
    return result;
  }

  int pos_m = 0;
  int controlWindow_m = CONTROL_WINDOW;
  double score_m = 0.0;
  std::atomic<unsigned int> currModel_m;
  std::atomic<unsigned int> prevModel_m;
  std::unique_ptr<SmoothingFilter> filter_m;

  double Kp = 0.6;
  double Kd = 0.4;
  double prev_error_m = 0.0;
  double time_m = 0;
  double integral_m = 0.0;

  double errIncreasePerSize_m;
  double errVariance_m;
  double prevTimeSpent_m = -1.0;
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

  // Smoothing filter
  if (prevModel_m != currModel_m) {
    filter_m->reset();
  }

  filter_m->update(time_spent);
  time_spent = filter_m->predict();

  // error deviation from the target
  double error = FRAME_DEADLINE - time_spent;

  // update adaptive threshold parameters
  updateErrIncreasePerSize(currModel_m, time_spent);

  // PID filter
  integral_m += error;
  double error_diff = prevModel_m != currModel_m ? 0.0 : (error - prev_error_m);
  time_m += 1;
  double error_hat = pid_filter(error, integral_m, error_diff);

  // Get thresholds
  // double theta_low = UP_SUM;    // (0.1 * FRAME_DEADLINE);
  // double theta_high = DOWN_SUM; // (0.05 * FRAME_DEADLINE);
  double theta_low, theta_high;
  getThresholds(theta_low, theta_high);

  // save previous model and error
  prevModel_m = unsigned(currModel_m);
  prev_error_m = error;

  // Control policy
  if (error_hat >= theta_low) {

    if (currModel_m < MAX_MODEL) {
      currModel_m = getAccModel(currModel_m + 1);
    }
    // std::cout << frameHdr.frameId << " : update up to : " << currModel_m <<
    // std::endl;
  } else if (error_hat <= (-1 * theta_high)) {
    if (currModel_m > MIN_MODEL) {
      currModel_m = getAccModel(currModel_m - 1);
    }
    // std::cout << frameHdr.frameId << " : update down to : " << currModel_m
    // << std::endl;
  }

  return 0;
}

int Controller::Impl::predict(std::shared_ptr<FrameObject> frameObject) {
  return currModel_m;
}

#endif /* __BASIC_CONTROLLER_H__ */
