#ifndef __COMMON_H__
#define __COMMON_H__

#include "json.hpp"
#include "message.h"
#include "queue.h"
#include <algorithm>

#define BUFF_SIZE 8192 // 4096 2048
#define MAX_FRAME_BUFFER_SIZE 30
#define FRAME_RATE 30
#define FRAME_ARRIVAL_TIME (1.0 / FRAME_RATE) * 1000
#define FRAME_DEADLINE 33
#define MAX_MODEL 18 // 15 for 16model server, 18 for others
#define MIN_MODEL 0
#define STARTING_MODEL 0

#define CONTROL_WINDOW 50
#define LOW_ON_TIME 35  // switch down if less or equal to X are on time
#define HIGH_ON_TIME 50 // switch up if more or equal to X are on time

#define DOWN_SUM 2.5    // switch down when average latency score is 5ms late
#define UP_SUM 5        // switch up when average latency score is 10ms early
#define LATE_EXP 1.25   // score exponent for late latency
#define ON_TIME_EXP 1.0 // score exponent for early latency

#define HISTORY_WEIGHT 0.0
// 1.0/4 // weight for the history  		Normal weight =
// (1-History weight)

typedef Queue<std::shared_ptr<Message_t>> MessageQueue_t;
typedef Queue<std::shared_ptr<FrameObject>> FrameQueue_t;
typedef Queue<std::shared_ptr<DetectionObject>> DetectionQueue_t;

// object that is returned by the server in which information on a detected
// object is stored
struct result_obj {
  unsigned int x, y, w, h;
  float prob;
  unsigned int obj_id;
};

// frame object that stores all the information about a frame
struct frame_obj {
  unsigned int frame_id;
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point server_received;
  unsigned int correct_model;
  unsigned int used_model;
  cv::Mat frame;
  std::vector<uchar> vec;
  double time_till_send;
  double time_after_send;
  double time_till_detection;
  double detection_time;
};

/* TODO: Move to a config file and parse it into a singleton config object */
const unsigned int n_height[19] = {64,  96,  128, 160, 192, 224, 256,
                                   288, 320, 352, 384, 416, 448, 480,
                                   512, 544, 576, 608, 640};
const unsigned int n_width[19] = {64,  96,  128, 160, 192, 224, 256,
                                  288, 320, 352, 384, 416, 448, 480,
                                  512, 544, 576, 608, 640};

const int coco_ids[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
                        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                        80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

#define JSON_VALUE_IF_PRESENT(jsonObject, key, type, defaultValue)             \
  jsonObject.find(key) != jsonObject.end() ? jsonObject[key].get<type>()       \
                                           : defaultValue
class ConfigManager {
public:
  // common config
  std::string serverHost;
  unsigned int serverPort;
  int networkShaping;
  std::string shapingFile;
  std::string videoPath;
  unsigned int maxFrames;
  unsigned int startingModel;
  unsigned int totalModels;
  std::string outputFile;
  std::string imageList;

  // darknet config
  std::string darknetDataDir;
  std::string darknetCfgFile;
  std::string darknetWeightFile;

  // config for basic controller
  double downSum;
  double lateExp;
  double upSum;
  double onTimeExp;
  double historyWeight;

  // config for MPC controller
  std::vector<double> modelsPredTime;
  std::vector<double> modelsAccuracy;
  std::vector<size_t> refFrameSize;
  double weightForSwitching;
  double weightForAccuracy;
  std::string glpkLpFile; // lp file for glpk solver
  int throughputHistoryLen;
  int solverInterval;

  ConfigManager() {}
  static ConfigManager *Default() {
    static ConfigManager *configManager = new ConfigManager();
    return configManager;
  }

  void readConfig(const std::string &fileName) {
    std::ifstream configFile(fileName);
    nlohmann::json jsonObject;
    try {
      configFile >> jsonObject;
      readCommonConfig(jsonObject);
      readBasicControllerConfig(jsonObject);
      readMPCConfig(jsonObject);
    } catch (std::exception e) {
      std::cerr << e.what() << std::endl;
      throw std::runtime_error("JSON Parse error");
    }
  }

private:
  void readCommonConfig(const nlohmann::json &jsonObject) {
    serverHost =
        JSON_VALUE_IF_PRESENT(jsonObject, "server_host", std::string, "");
    serverPort =
        JSON_VALUE_IF_PRESENT(jsonObject, "server_port", unsigned int, 0);
    videoPath =
        JSON_VALUE_IF_PRESENT(jsonObject, "video_path", std::string, "");
    networkShaping =
        JSON_VALUE_IF_PRESENT(jsonObject, "network_shaping", int, 0);
    shapingFile =
        JSON_VALUE_IF_PRESENT(jsonObject, "shaping_file", std::string, "");
    maxFrames = JSON_VALUE_IF_PRESENT(jsonObject, "max_frames", unsigned int,
                                      UINT_MAX - 1);
    startingModel = JSON_VALUE_IF_PRESENT(jsonObject, "starting_model",
                                          unsigned int, STARTING_MODEL);
    totalModels = JSON_VALUE_IF_PRESENT(jsonObject, "total_models",
                                        unsigned int, MAX_MODEL + 1);
    outputFile = JSON_VALUE_IF_PRESENT(jsonObject, "output_file", std::string,
                                       "output.json");
    imageList =
        JSON_VALUE_IF_PRESENT(jsonObject, "image_list", std::string, "");

    darknetDataDir =
        JSON_VALUE_IF_PRESENT(jsonObject, "darknet_data_dir", std::string, "");
    darknetCfgFile =
        JSON_VALUE_IF_PRESENT(jsonObject, "darknet_cfg_file", std::string, "");
    darknetWeightFile = JSON_VALUE_IF_PRESENT(jsonObject, "darknet_weight_file",
                                              std::string, "");
  }

  void readBasicControllerConfig(const nlohmann::json &jsonObject) {
    downSum = JSON_VALUE_IF_PRESENT(jsonObject, "down_sum", double, DOWN_SUM);
    lateExp = JSON_VALUE_IF_PRESENT(jsonObject, "late_exp", double, LATE_EXP);
    upSum = JSON_VALUE_IF_PRESENT(jsonObject, "up_sum", double, UP_SUM);
    onTimeExp =
        JSON_VALUE_IF_PRESENT(jsonObject, "on_time_exp", double, ON_TIME_EXP);
    historyWeight = JSON_VALUE_IF_PRESENT(jsonObject, "history_weight", double,
                                          historyWeight);
  }

  void readMPCConfig(const nlohmann::json &jsonObject) {
    modelsPredTime =
        JSON_VALUE_IF_PRESENT(jsonObject, "models_prediction_time",
                              std::vector<double>, std::vector<double>());
    modelsAccuracy =
        JSON_VALUE_IF_PRESENT(jsonObject, "models_accuracy",
                              std::vector<double>, std::vector<double>());
    refFrameSize =
        JSON_VALUE_IF_PRESENT(jsonObject, "reference_frame_sizes",
                              std::vector<size_t>, std::vector<size_t>());
    weightForAccuracy =
        JSON_VALUE_IF_PRESENT(jsonObject, "weight_for_accuracy", double, 1.0);
    weightForSwitching =
        JSON_VALUE_IF_PRESENT(jsonObject, "weight_for_switching", double, 0.0);

    // GLPK config
    glpkLpFile =
        JSON_VALUE_IF_PRESENT(jsonObject, "glpk_lp_file", std::string, "");
    throughputHistoryLen =
        JSON_VALUE_IF_PRESENT(jsonObject, "throughput_history_len", int, 5);
    solverInterval =
        JSON_VALUE_IF_PRESENT(jsonObject, "solver_interval", int, 1);
  }
};

/* Utility functions will be defined here. */
namespace {
double getRTT(double throughput, double modelTime, double netSize,
              double lastEnd2Endtime = 0.0) {

  double networkTime = (netSize / throughput) * 8;
  double waitingTime =
      std::max((lastEnd2Endtime - FRAME_ARRIVAL_TIME - networkTime), 0.0);
  double rtt = networkTime + modelTime + waitingTime;
  return rtt; // ms
}

} // namespace

#endif /* __COMMON_H__ */
