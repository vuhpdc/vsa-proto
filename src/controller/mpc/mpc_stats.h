#ifndef __MPC_STATS_H__
#define __MPC_STATS_H__
#include "common.h"

#define MAX_STATS 5000

struct StatsData {
  unsigned int frameID;
  unsigned int desiredModel;
  unsigned int usedModel;
  double networkTime;
  double modelPredTime;
  double endToEndTime;
  size_t requestSize;
  size_t responseSize;
  size_t netSize;
  double throughput;
};

class MPCStats {
public:
  MPCStats() {
    totalModels_m = ConfigManager::Default()->totalModels;
    measured_m = std::make_unique<std::vector<StatsData>>(MAX_STATS);
    predicted_m = std::make_unique<std::vector<std::vector<StatsData>>>(
        MAX_STATS, std::vector<StatsData>(totalModels_m));
  }

  void print() {
    ofstream _cout("mpc.stats");
    auto totalStats = min(totalMeasuredStats_m, totalPredictedStats_m);
    auto solverInterval = ConfigManager::Default()->solverInterval;
    for (auto i = 0; i < totalStats; i++) {
      auto &measuredStats = (*measured_m)[i];
      assert(measuredStats.frameID == i);

      _cout << "MPCStats:Measured"
            << ",Frame:" << measuredStats.frameID
            << ",EndToTime:" << measuredStats.endToEndTime
            << ",NetworkTime:" << measuredStats.networkTime
            << ",ModelPredTime:" << measuredStats.modelPredTime
            << ",NetworkSize:" << measuredStats.netSize
            << ",DesiredModel:" << measuredStats.desiredModel
            << ",UsedModel:" << measuredStats.usedModel << "\n";

      if (i != 0 && (i % solverInterval) == 0) {
        auto &desiredPredictedStats =
            (*predicted_m)[i][measuredStats.desiredModel];
        auto &usedPredictedStats = (*predicted_m)[i][measuredStats.usedModel];
        // assert(desiredPredictedStats.frameID == i);
        // assert(usedPredictedStats.frameID == i);

        _cout << "MPCStats:DesiredPredicted"
              << ",Frame:" << desiredPredictedStats.frameID
              << ",EndToTime:" << desiredPredictedStats.endToEndTime
              << ",NetworkTime:" << desiredPredictedStats.networkTime
              << ",ModelPredTime:" << desiredPredictedStats.modelPredTime
              << ",NetworkSize:" << desiredPredictedStats.netSize
              << ",DesiredModel:" << desiredPredictedStats.desiredModel
              << ",Throughput:" << desiredPredictedStats.throughput << "\n";

        _cout << "MPCStats:UsedPredicted"
              << ",Frame:" << usedPredictedStats.frameID
              << ",EndToTime:" << usedPredictedStats.endToEndTime
              << ",NetworkTime:" << usedPredictedStats.networkTime
              << ",ModelPredTime:" << usedPredictedStats.modelPredTime
              << ",NetworkSize:" << usedPredictedStats.netSize
              << ",DesiredModel:" << desiredPredictedStats.desiredModel
              << ",Throughput:" << usedPredictedStats.throughput << "\n";

        _cout << "MPCStats:Variance"
              << ",Frame:" << measuredStats.frameID
              << ",NetworkTime:"
              << (measuredStats.networkTime - desiredPredictedStats.networkTime)
              << ",ModelPredTime:"
              << (measuredStats.modelPredTime - usedPredictedStats.modelPredTime)
              << "\n";
        }
      }
  }

  void update(const unsigned int frameID, const unsigned int desiredModel,
              const unsigned int usedModel, const double networkTime,
              const double modelPredTime, const size_t requestSize,
              const size_t responseSize) {
    assert(frameID >= totalMeasuredStats_m);

    auto &stats = (*measured_m)[frameID];
    stats.frameID = frameID;
    stats.desiredModel = desiredModel;
    stats.usedModel = usedModel;
    stats.networkTime = networkTime;
    stats.modelPredTime = modelPredTime;
    stats.endToEndTime = (networkTime + modelPredTime);
    stats.requestSize = requestSize;
    stats.responseSize = responseSize;
    stats.netSize = requestSize + responseSize;

    totalMeasuredStats_m = frameID;
  }

  void predict(const unsigned int frameID, const unsigned int desiredModel,
               const std::vector<double> &throughputHorizon,
               const std::vector<std::vector<double>> &modelTimeHorizon,
               const std::vector<std::vector<size_t>> &netSizeHorizon) {

    assert(frameID >= totalPredictedStats_m);

    auto horizon = 0;
    for (auto model = 0; model < totalModels_m; ++model) {
      auto &stats = (*predicted_m)[frameID][model];
      stats.frameID = frameID;
      stats.desiredModel = desiredModel;
      stats.endToEndTime =
          getRTT(throughputHorizon[horizon], modelTimeHorizon[horizon][model],
                 netSizeHorizon[horizon][model]);
      stats.modelPredTime = modelTimeHorizon[horizon][model];
      stats.networkTime = (stats.endToEndTime - stats.modelPredTime);
      stats.netSize = netSizeHorizon[horizon][model];
      stats.throughput = throughputHorizon[horizon];
    }

    totalPredictedStats_m = frameID;
  }

private:
  unsigned int totalModels_m;
  unsigned int totalPredictedStats_m = 0;
  unsigned int totalMeasuredStats_m = 0;
  std::unique_ptr<std::vector<StatsData>> measured_m;
  std::unique_ptr<std::vector<std::vector<StatsData>>> predicted_m;
};
#endif /* __MPC_STATS_H__ */
