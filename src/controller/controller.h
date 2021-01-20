#ifndef __CONTROLLER_H__
#define __CONTROLLER_H__

#include "message.h"
#include <atomic>
#include <memory>
class Controller {
public:
  class Impl;
  explicit Controller(Impl *impl) : _impl(impl) {}
  int update(std::shared_ptr<DetectionObject> &);
  int predict(std::shared_ptr<FrameObject> &) const;

private:
  std::unique_ptr<Impl> _impl;
  std::atomic_uint iter_m{0};
};

#endif /* __CONTROLLER_H__ */
