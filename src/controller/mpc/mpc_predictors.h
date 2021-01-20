#ifndef __MPC_PROVIDERS_H__
#define __MPC_PROVIDERS_H__
#include "message.h"
#include <assert.h>
#include <atomic>
#include <vector>

/**
 * @brief Basic throughput predictor. This is a goodput at application level.
 * @todo: We need a matured and correct approximate throughput measurement
 * and predictor. Not this TOY predictor.
 *
 */
class ThroughputPredictor {
public:
  explicit ThroughputPredictor(unsigned int horizon,
                               unsigned int historySampleLen,
                               unsigned int stalenessGap)
      : horizon_m(horizon), historySampleLen_m(historySampleLen), stalenessGap_m(stalenessGap) {
    measuredTroughput_m.resize(historySampleLen_m);
  }
  void update(unsigned int iter, double networkTime, size_t requestSize,
              size_t responseSize) {
    assert(iter == iter_m);

    // compute
    auto total_size = requestSize + responseSize;
    double throughput = total_size / networkTime * 8; // kbps

    // std::cout << "Iter: " << iter << "\n"
    //    "\tNetworkTime: " << networkTime <<
    //    "\tRequestSize: " << requestSize <<
    //    "\t ResponseSize: " << responseSize <<
    //    "\t Throughput: " << throughput << "\n";

    // measuredTroughput_m.push_back(throughput);
    measuredTroughput_m[index_m % historySampleLen_m] = throughput;
    ++index_m;
    ++iter_m;
  }

  std::vector<double> predict(unsigned int iter) const {
    // assert(iter == iter_m); // We relax predict to use some stale update information. And it is controlled through `stalenessGap'
    assert((iter - iter_m) <= stalenessGap_m);

    // TODO: Use harmonic mean
    unsigned int endIndex;
    if (iter_m < historySampleLen_m) {
      endIndex = index_m;
    } else {
      endIndex = historySampleLen_m;
    }

    double sum = 0.0;
    for (auto i = 0; i < endIndex; ++i) {
      sum += double(1.0 / measuredTroughput_m[i]);
    }
    auto predictedThroughput = endIndex / sum;

    std::vector<double> predThroughput(horizon_m);
    for (auto i = 0; i < horizon_m; ++i) {
      predThroughput[i] = predictedThroughput;
    }
    return predThroughput;
  }

private:
  std::atomic_uint iter_m{0};
  std::vector<double> measuredTroughput_m;
  std::atomic_int index_m{0};
  unsigned int horizon_m;
  unsigned int historySampleLen_m;
  unsigned int stalenessGap_m;
};

/**
 * @brief Predict model execution time on the server side.
 * Use a slowdown factor, similar to the ALERT paper.
 *
 */
class ModelTimePredictor {
public:
  explicit ModelTimePredictor(unsigned int horizon, unsigned int totalModels,
                              std::vector<double> profiledTime,
                              unsigned int stalenessGap)
      : profiledTime_m(profiledTime), totalModels_m(totalModels),
        horizon_m(horizon), stalenessGap_m(stalenessGap) {}

  void update(unsigned int iter, unsigned int usedCorrectModel,
              double modelTime) {
    assert(iter == iter_m);
    double slowDownFactor = modelTime / profiledTime_m[usedCorrectModel];
    update(slowDownFactor);
    ++iter_m;
  }

  std::vector<std::vector<double>> predict(unsigned int iter) const {
    // assert(iter == iter_m);
    assert((iter - iter_m) <= stalenessGap_m);
    std::vector<std::vector<double>> modelPredTime(horizon_m);
    auto slowDownFactors = predict();
    for (auto i = 0; i < horizon_m; ++i) {
      modelPredTime[i].resize(totalModels_m);
      for (auto j = 0; j < totalModels_m; ++j) {
        modelPredTime[i][j] = slowDownFactors[i] * profiledTime_m[j];
      }
    }
    return modelPredTime;
  }

private:
  void update(double slowDownFactor) {
    if (slowDownFactor_m == -1.0) {
      slowDownFactor_m = slowDownFactor;
    } else {
      slowDownFactor_m = movingAvgWeight_m * slowDownFactor_m +
                         (1.0 - movingAvgWeight_m) * slowDownFactor;
    }
  }

  std::vector<double> predict() const {
    std::vector<double> slowDownFactors(horizon_m);
    for (auto i = 0; i < horizon_m; ++i) {
      slowDownFactors[i] = slowDownFactor_m;
    }
    return slowDownFactors;
  }

  const double movingAvgWeight_m = 0.75;
  double slowDownFactor_m = 1.0;
  std::vector<double> profiledTime_m;
  std::atomic_uint iter_m{0};
  const unsigned int totalModels_m;
  unsigned int horizon_m;
  unsigned int stalenessGap_m;
};

/**
 * @brief Predict data size on the network
 *          1. the encoded size of the image. Note that encoded size of the same
 *              size frame might change depending on the content of the frame.
 *          2. Response size changes depending upon on the number of objectes
 *              detected in the frame.
 *
 */
class NetworkSizePredictor {

public:
  explicit NetworkSizePredictor(unsigned int horizon, unsigned int totalModels,
                                std::vector<size_t> refFrameSize,
                                unsigned int stalenessGap)
      : horizon_m(horizon), refFrameSize_m(refFrameSize),
        totalModels_m(totalModels), stalenessGap_m(stalenessGap) {
    maxFrameSizeSeen_m.resize(totalModels, 0.0);
  }

  void update(unsigned int iter, unsigned int desiredModel, size_t requestSize,
              size_t responseSize) {
    assert(iter == iter_m);
    auto frameEncodedSize =
        requestSize - sizeof(MessageHeader_t) - sizeof(FrameHeader_t);
    frameSizeFactor_m = (double)frameEncodedSize / refFrameSize_m[desiredModel];
    responseSize_m = responseSize;

    // if (maxFrameSizeSeen_m[desiredModel] < frameEncodedSize) {
    //   maxFrameSizeSeen_m[desiredModel] = frameEncodedSize;
    // }

    ++iter_m;
  }

  std::vector<std::vector<size_t>> predict(unsigned int iter,
                                           cv::Mat &srcFrame) const {
    // assert(iter == iter_m);
    assert((iter - iter_m) <= stalenessGap_m);
    std::vector<size_t> requestSizes(totalModels_m);
    for (auto i = 0; i < totalModels_m; ++i) {
      // auto frameSizeFactor = getFrameSizeFactor(srcFrame, i);
      // requestSizes[i] = frameSizeFactor * refFrameSize_m[i];
      requestSizes[i] = frameSizeFactor_m * refFrameSize_m[i];
      if (requestSizes[i] < maxFrameSizeSeen_m[i]) {
          requestSizes[i] = maxFrameSizeSeen_m[i];
      }
    }
    return _predict(requestSizes);
  }
  std::vector<std::vector<size_t>> predict(unsigned int iter) const {
    assert(iter == iter_m);
    // use encoded size of previous by assuming it remains the same for this
    // frame
    // return _predict(frameSizeFactor_m);
    return  std::vector<std::vector<size_t>>();
  }

private:
  std::vector<std::vector<size_t>>
  _predict(const std::vector<size_t> &requestSizes) const {
  // _predict(const double &frameSizeFactor) const {

    std::vector<std::vector<size_t>> netSizePred(horizon_m);
    // auto frameSizeFactors = predictFrameSizeFactors(frameSizeFactor);

    for (auto i = 0; i < horizon_m; ++i) {
      netSizePred[i].resize(totalModels_m);
      for (auto j = 0; j < totalModels_m; ++j) {
        // auto requestSize = frameSizeFactors[i] * refFrameSize_m[j];
        auto requestSize = requestSizes[j];
        requestSize += (sizeof(MessageHeader_t) + sizeof(FrameHeader_t));
        auto responseSize = responseSize_m; // use same response size
        netSizePred[i][j] = requestSize + responseSize;
      }
    }
    return netSizePred;
  }

  double getFrameSizeFactor(const cv::Mat &srcFrame, unsigned int inputSize) const {
    // Resize src frame to smallest model input size. This is to do it fast.
    cv::Mat dstFrame;
    // unsigned int inputSize = 0;
    cv::resize(srcFrame, dstFrame,
               cv::Size(n_width[inputSize], n_height[inputSize]), 1, 1,
               cv::INTER_NEAREST);

    // Encode frame
    std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};
    std::vector<unsigned char> encodedVec;
    cv::imencode(".jpg", dstFrame, encodedVec, compression_params);

    auto frameSizeFactor =
        (double)encodedVec.size() / refFrameSize_m[inputSize];

    return frameSizeFactor;
  }

  void update(double frameSizeFactor) {
    frameSizeFactor_m = movingAvgWeight_m * frameSizeFactor_m +
                        (1.0 - movingAvgWeight_m) * frameSizeFactor;
  }

  std::vector<double>
  predictFrameSizeFactors(const double &frameSizeFactor) const {
    std::vector<double> frameSizeFactors(horizon_m);
    for (auto i = 0; i < horizon_m; ++i) {
      frameSizeFactors[i] =
          frameSizeFactor; // keep same factor for the entire horizon
    }
    return frameSizeFactors;
  }

  const double movingAvgWeight_m = 0.5;
  unsigned int horizon_m;
  std::atomic_uint iter_m{0};
  double frameSizeFactor_m = 1.0;
  std::vector<size_t> refFrameSize_m;
  std::vector<size_t> maxFrameSizeSeen_m;
  size_t responseSize_m;
  unsigned int totalModels_m;
  unsigned int stalenessGap_m;
};

#endif /* __MPC_PROVIDERS_H__ */
