#ifndef __MPC_CONTROLLER_H__
#define __MPC_CONTROLLER_H__
#include "controller.h"
#include "mpc_predictors.h"
#include "mpc_solver.h"
#include "mpc_stats.h"
#include <assert.h>

#define MPC_HORIZON 5
class Controller::Impl {
public:
  Impl(unsigned int currModel) : currModel_m(currModel) {
    std::cout << "MPC Controller" << std::endl;
    auto totalModels = ConfigManager::Default()->totalModels;
    throughputPredictor_m = std::make_unique<ThroughputPredictor>(
        MPC_HORIZON, ConfigManager::Default()->throughputHistoryLen,
        stalenessGap_m);
    modelTimePredictor_m = std::make_unique<ModelTimePredictor>(
        MPC_HORIZON, totalModels, ConfigManager::Default()->modelsPredTime,
        stalenessGap_m);
    networkSizePredictor_m = std::make_unique<NetworkSizePredictor>(
        MPC_HORIZON, totalModels, ConfigManager::Default()->refFrameSize,
        stalenessGap_m);

    SOLVER_INTERVAL = ConfigManager::Default()->solverInterval;

    auto solver_ptr = Solver::createInstance(
        MPC_HORIZON, totalModels, ConfigManager::Default()->modelsAccuracy,
        STARTING_MODEL, ConfigManager::Default()->weightForAccuracy,
        ConfigManager::Default()->weightForSwitching);
    mpcSolver_m.reset(solver_ptr);
  }

  ~Impl() { stats_m.print(); }

  int update(std::shared_ptr<DetectionObject> &);
  int predict(std::shared_ptr<FrameObject> &);
  int runSolver() { return solver_iter_m == SOLVER_INTERVAL; }

  int predictDefault() {
    predEndToEndTime_m.push_back(0.0);
    return currModel_m;
  }
  const unsigned int stalenessGap_m = 8;

private:
  void extractTime(const std::shared_ptr<DetectionObject> &detectionObject,
                   double &networkTime, double &modelPredTime) const {
    auto &detectionHdr = detectionObject->detectionHdr;
    auto &frameHdr = detectionHdr.frameHdr;
    modelPredTime =
        TIME_US_TO_MS(frameHdr.serverSendTime - frameHdr.serverRecvTime);
    auto endToEndTime =
        TIME_US_TO_MS(frameHdr.clientRecvTime - frameHdr.clientSendTime);
    networkTime = endToEndTime - modelPredTime;
  }

  void extractSize(const std::shared_ptr<DetectionObject> &detectionObject,
                   size_t &requestSize, size_t &responseSize) const {
    auto &detectionHdr = detectionObject->detectionHdr;
    auto &frameHdr = detectionHdr.frameHdr;
    requestSize = frameHdr.serializedSize + sizeof(MessageHeader_t);
    responseSize = detectionHdr.serializedSize + sizeof(MessageHeader_t);

    // double throughput =
    //     ((frameNetSize + detectionNetSize) / networkTime_ms) * 8; // kbps
  }

  unsigned int currModel_m;
  unsigned int lastDesiredModel_m;
  unsigned int lastUsedModel_m;
  std::unique_ptr<ThroughputPredictor> throughputPredictor_m;
  std::unique_ptr<ModelTimePredictor> modelTimePredictor_m;
  std::unique_ptr<NetworkSizePredictor> networkSizePredictor_m;
  std::vector<double> predEndToEndTime_m;
  std::unique_ptr<Solver> mpcSolver_m;
  int SOLVER_INTERVAL;
  std::atomic_uint solver_iter_m{0};
  std::mutex mutex_m;
  MPCStats stats_m;
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
  // while (frameObject->frameHdr.frameId != iter_m) {
  // TODO: We wait for earlier frame to finish, so that feedback can be
  // incorporated for this frame.
  // return -1;
  //  usleep(1000);
  //}

  // Measurements might not be available yet to predict
  if (frameObject->frameHdr.frameId == 0 || iter_m == 0) {
    return _impl->predictDefault();
  }

  // check if it's solver time and all feedbacks (updates) are received for
  // previous frames
  while (_impl->runSolver() &&
         (frameObject->frameHdr.frameId - iter_m) > _impl->stalenessGap_m) {
    usleep(1000);
  }

  return _impl->predict(frameObject);
}

int Controller::Impl::update(
    std::shared_ptr<DetectionObject> &detectionObject) {

  auto frameId = detectionObject->detectionHdr.frameHdr.frameId;
  auto desiredModel = detectionObject->detectionHdr.frameHdr.correctModel;
  auto usedModel = detectionObject->detectionHdr.usedModel;

  // Extract feedback values
  double networkTime, modelPredTime;
  extractTime(detectionObject, networkTime, modelPredTime);
  size_t requestSize, responseSize;
  extractSize(detectionObject, requestSize, responseSize);

  // collect measured stats
  stats_m.update(frameId, desiredModel, usedModel, networkTime, modelPredTime,
                 requestSize, responseSize);

  // std::cout << "Frame:" << frameId
  //           << ",EndToTime:" << (networkTime + modelPredTime)
  //           << ",NetworkTime:" << networkTime
  //           << ",ModelPredTime:" << modelPredTime
  //           << ",RequestSize:" << requestSize
  //           << ",ResponseSize:" << responseSize << "\n";

  // Update system measured variables
  throughputPredictor_m->update(frameId, networkTime, requestSize,
                                responseSize);
  modelTimePredictor_m->update(frameId, usedModel, modelPredTime);
  networkSizePredictor_m->update(frameId, desiredModel, requestSize,
                                 responseSize);

  // update models
  lastDesiredModel_m = desiredModel;
  lastUsedModel_m = usedModel;

  // reset end to end time
  std::unique_lock<std::mutex> lock(mutex_m);
  // std::cout << "Update predEndToEndTime_m: " << frameId << std::endl;
  predEndToEndTime_m[frameId] = 0.0;
  return -1;
}

int Controller::Impl::predict(std::shared_ptr<FrameObject> &frameObject) {

  // Predict control horizon
  auto throughputPredHorizon =
      throughputPredictor_m->predict(frameObject->frameHdr.frameId);
  auto modelTimePredHorizon =
      modelTimePredictor_m->predict(frameObject->frameHdr.frameId);
  // auto networkSizePredHorizon =
  //     networkSizePredictor_m->predict(frameObject->frameHdr.frameId);
  auto networkSizePredHorizon = networkSizePredictor_m->predict(
      frameObject->frameHdr.frameId, frameObject->frameMat);

  std::vector<double> solverEnd2EndTimePred;
  std::unique_lock<std::mutex> lock(mutex_m);
  assert(frameObject->frameHdr.frameId == predEndToEndTime_m.size());
  auto lastEndToEndTime = predEndToEndTime_m[frameObject->frameHdr.frameId - 1];
  lock.unlock();
  // std::cout << "Previous lastEndToEndTime: " << frameObject->frameHdr.frameId << " " << lastEndToEndTime << std::endl;
  if (solver_iter_m == SOLVER_INTERVAL) {
    auto modelsHorizon = mpcSolver_m->solve(
        throughputPredHorizon, modelTimePredHorizon, networkSizePredHorizon,
        ConfigManager::Default()->modelsAccuracy[lastUsedModel_m],
        lastEndToEndTime, solverEnd2EndTimePred);

    if (modelsHorizon.empty()) {
      // TODO: Strategy is to use previous input size so that we do not abruptly
      // switch input sizes. But this may to lead to long latency if input_size
      // is large.
      std::cout << "WARNING: mpc solver: no solution for "
                << frameObject->frameHdr.frameId << std::endl;

    } else {
      currModel_m = modelsHorizon[0];
    }
    solver_iter_m = 1;

    // collect predict stats
    stats_m.predict(frameObject->frameHdr.frameId, currModel_m,
                    throughputPredHorizon, modelTimePredHorizon,
                    networkSizePredHorizon);
  } else {
    ++solver_iter_m;
  }

  lastEndToEndTime =
      getRTT(throughputPredHorizon[0], modelTimePredHorizon[0][currModel_m],
             networkSizePredHorizon[0][currModel_m], lastEndToEndTime);
  lock.lock();
  // std::cout << "Current lastEndToEndTime: " << frameObject->frameHdr.frameId << " " << lastEndToEndTime << std::endl;
  predEndToEndTime_m.push_back(lastEndToEndTime);
  if (solverEnd2EndTimePred.size() != 0 && solverEnd2EndTimePred[0] != lastEndToEndTime) {
     std::cout << "Waiting time difference: " << (solverEnd2EndTimePred[0] - lastEndToEndTime)
        << " " << solverEnd2EndTimePred[0] << " " << lastEndToEndTime << std::endl;
  }
  // assert (solverEnd2EndTimePred.size() == 0 ||
  //    (solverEnd2EndTimePred.size() != 0 && solverEnd2EndTimePred[0] == lastEndToEndTime));
  return currModel_m;
}

#endif /* __MPC_CONTROLLER_H__ */
