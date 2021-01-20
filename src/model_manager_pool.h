#ifndef __MODEL_MANAGER_POOL_H__
#define __MODEL_MANAGER_POOL_H__

#include "common.h"
#include "message.h"
#include "network.h"
#include "yolo_v2_class.hpp"
#include <atomic>
#include <list>
#include <thread>
#include <unordered_map>

#define POOL_SIZE 8
#define N_GPUS 2
#define NFRAMES 3

// object that stores all components of detector model
struct detector_gpu_t {
  network net;
  image images[NFRAMES];
  float *avg;
  float *predictions[NFRAMES];
  int demo_index;
  unsigned int *track_id;
};

class ModelInstance {
public:
  ModelInstance(unsigned int initModel, unsigned int gpu_id)
      : currModel_m(initModel), desiredModel_m(initModel), gpu_id_m(gpu_id) {
    detector_m = std::make_unique<Detector>(
        ConfigManager::Default()->darknetCfgFile,
        ConfigManager::Default()->darknetWeightFile, gpu_id_m);
    resize();
  }

  ModelInstance(const ModelInstance &) = delete;
  ModelInstance &operator=(const ModelInstance &) = delete;

  bool reserveModel() {
    // reserve model for use
    std::unique_lock<std::mutex> lock(mutex_m);
    if (inUse || resizeFlag)
      return false;
    inUse = true;
    return true;
  }

  std::vector<DetectionObject_t> detect(const cv::Mat &frame) const {
    auto ret = detector_m->detect(frame);
    return ret;
  }

  void releaseModel() {
    std::unique_lock<std::mutex> lock(mutex_m);
    assert(inUse == true);
    inUse = false;
  }

  void resize() {
    if (resizeFlag) {
      while (inUse) {
        usleep(1000); // TODO: replace all sleep calls with condittional wait
      }
      detector_gpu_t &detector_gpu =
          *static_cast<detector_gpu_t *>(detector_m->detector_gpu_ptr.get());
      network &net = detector_gpu.net;
      resize_network(&net, n_height[desiredModel_m], n_width[desiredModel_m]);
      currModel_m = desiredModel_m;
      resizeFlag = false;
    }
  }

  void setDesiredModel(unsigned int desiredModel) {
    resizeFlag = true;
    desiredModel_m = desiredModel;
  }

  unsigned int getCurrModel() const { return currModel_m; }

private:
  std::unique_ptr<Detector> detector_m;
  std::atomic_bool inUse{false};
  std::atomic_bool resizeFlag{true};
  unsigned int currModel_m;
  unsigned int desiredModel_m;
  unsigned int gpu_id_m;
  std::mutex mutex_m;
};

typedef Queue<std::shared_ptr<ModelInstance>> ModelInstanceQueue_t;
typedef std::pair<unsigned int, std::shared_ptr<ModelInstance>>
    ModelKeyValuePair_t;
typedef typename std::list<ModelKeyValuePair_t>::iterator ModelListIterator_t;

class ModelManager {
public:
  explicit ModelManager(int currModel) {
    // totalModels_m = ConfigManager::Default()->totalModels;
    totalModels_m = MAX_MODEL + 1;
    init();
  }

  std::vector<DetectionObject_t>
  detect(cv::Mat &frame, unsigned int desiredModel, unsigned int &usedModel) {
    auto start_time = EnvTime::Default()->NowMicros();
    std::unique_lock<std::mutex> lock(mutex_m);

    auto it = modelPool_m.find(desiredModel);
    std::shared_ptr<ModelInstance> modelInstance;
    if (it ==
        modelPool_m.end()) { // Model not found, probably a new model request
      // evict the least recently used model and load with this desired model
      auto lastModelInstance = evictLRUModel();
      lastModelInstance->setDesiredModel(desiredModel);
      putModelInPool(lastModelInstance, desiredModel);
      resizeModelQueue.enqueue(lastModelInstance);

      // Find next best model but don't put at the front
      modelInstance = findNextBestModel(desiredModel, true);
    } else {
      // Found the desired model in the list. Move it to the front of the list.
      modelList_m.splice(modelList_m.begin(), modelList_m, it->second);
      modelInstance = it->second->second;
    }

    // try to reserve the model. Model cannot be reserved if it's inUse or
    // resizing.
    while (!modelInstance->reserveModel()) {
      modelInstance = findNextBestModel(desiredModel, true);
    }
    usedModel = modelInstance->getCurrModel();
    lock.unlock();

    auto end_time = EnvTime::Default()->NowMicros();
    // std::cout << "Model retrieval time: " <<
    // TIME_US_TO_MS(end_time-start_time) << "\n";

    start_time = EnvTime::Default()->NowMicros();
    auto ret = modelInstance->detect(frame);
    end_time = EnvTime::Default()->NowMicros();
    // std::cout << "Model detection time: " <<
    // TIME_US_TO_MS(end_time-start_time) << "\n";
    modelInstance->releaseModel();
    return ret;
  }

  ~ModelManager() {
    modelLoaderState = STOP;
    modelLoader_m.join();
  }

private:
  void init() {
    // load pool of models on GPUs.
    for (auto i = 0, currModel = 0;
         (i < POOL_SIZE && currModel < totalModels_m); ++i, ++currModel) {
      auto gpuId = currModel % N_GPUS;
      auto modelInstance = std::make_shared<ModelInstance>(currModel, gpuId);
      putModelInPool(modelInstance, currModel);
    }

    // start model loader
    modelLoaderState = START;
    modelLoader_m = std::thread(&ModelManager::model_loader, this);
  }

  void putModelInPool(std::shared_ptr<ModelInstance> &modelInstance,
                      unsigned int key) {
    // Note that the caller should already take the lock on the datastructure
    auto modelKeyValuePair = ModelKeyValuePair_t(key, modelInstance);
    modelList_m.push_front(modelKeyValuePair);
    modelPool_m[key] = modelList_m.begin();
  }

  std::shared_ptr<ModelInstance> evictLRUModel() {
    // Note that the caller should already take the lock on the datastructure
    auto lastModelPair = modelList_m.back();
    modelList_m.pop_back();
    modelPool_m.erase(lastModelPair.first);
    return lastModelPair.second;
  }

  std::shared_ptr<ModelInstance> findModel(unsigned int desiredModel) {
    auto it = modelPool_m.find(desiredModel);
    if (it != modelPool_m.end()) {
      return it->second->second;
    }
    return nullptr;
  }

  std::shared_ptr<ModelInstance> findNextBestModel(unsigned int desiredModel,
                                                   bool blocking) {
    // Note that caller should have already taken the lock on the data
    // structures.
    while (blocking) {
      // std::unique_lock<std::mutex> lock(mutex_m);
      auto modelInstance = findNextBestModel(desiredModel);
      if (modelInstance != nullptr) {
        return modelInstance;
      } else {
        // lock.unlock();
        usleep(1000);
      }
    }
  }

  std::shared_ptr<ModelInstance> findNextBestModel(unsigned int desiredModel) {
    // Note that the caller should already take the lock on the data structure
    for (auto i = 1; i < totalModels_m; ++i) {
      auto prev_model = desiredModel - i;
      auto next_model = desiredModel + i;
      if (prev_model >= 0) { // Look for a smaller model first, so that
                             // downscale is performed on the frame.
        auto modelInstance = findModel(prev_model);
        if (modelInstance != nullptr) {
          return modelInstance;
        }
      }

      // TODO: We should not select the bigger model, not just upscale affect
      // the accuracy but also increases the latency
      // if (next_model < totalModels_m) { // Next Look for a bigger model, here
      //                                   // upscale is performed on the frame.
      //   auto modelInstance = findModel(next_model);
      //   if (modelInstance != nullptr) {
      //     return modelInstance;
      //   }
      // }
    }
    return nullptr;
  }

  void model_loader() {
    while (modelLoaderState != STOP) {
      std::shared_ptr<ModelInstance> modelInstance;
      if (resizeModelQueue.dequeue(modelInstance, 1)) {
        modelInstance->resize();
      }
    }
  }

  enum ModelLoaderState { START = 0, LOAD = 1, STOP = 2 };
  std::mutex mutex_m;
  std::atomic<ModelLoaderState> modelLoaderState;
  std::thread modelLoader_m;
  unsigned int totalModels_m;
  /* Maintains a list of model and its instance */
  std::list<ModelKeyValuePair_t> modelList_m;
  /* Maintains a map of model and its position in the model list */
  std::unordered_map<unsigned int, ModelListIterator_t> modelPool_m;

  ModelInstanceQueue_t resizeModelQueue;
};
#endif /* __MODEL_MANAGER_POOL_H__ */
