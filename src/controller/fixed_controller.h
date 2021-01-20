#ifndef __FIXED_CONTROLLER_H__
#define __FIXED_CONTROLLER_H__

#include "controller.h"

class Controller::Impl {
public:
  Impl(unsigned int currModel) : currModel_m(currModel) {
    std::cout << "Fixed Controller" << std::endl;
  }
  int update(std::shared_ptr<DetectionObject>);
  int predict(std::shared_ptr<FrameObject>);

private:
  unsigned int currModel_m;
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
  return _impl->predict(frameObject);
}

int Controller::Impl::update(std::shared_ptr<DetectionObject> detectionObject) {
  return 0; // nop
}

int Controller::Impl::predict(std::shared_ptr<FrameObject> frameObject) {
  return currModel_m;
}

#endif /* __FIXED_CONTROLLER_H__ */