#include "network.h"
#include "yolo_v2_class.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>
#include <vector>

#include "common.h"
#include "env_time.h"

using namespace cv;

#define NFRAMES 3
struct detector_gpu_t {
  network net;
  image images[NFRAMES];
  float *avg;
  float *predictions[NFRAMES];
  int demo_index;
  unsigned int *track_id;
};

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec,
                std::vector<std::string> obj_names, double multiplier) {
  for (auto &i : result_vec) {
    int x = (int)(i.x / multiplier);
    int w = (int)(i.w / multiplier);
    int y = (int)(i.y / multiplier);
    int h = (int)(i.h / multiplier);
    cv::rectangle(mat_img, cv::Point(x, y), cv::Point(x + w, y + h),
                  cv::Scalar(255, 178, 50), 3);
    if (obj_names.size() > i.obj_id) {
      std::string label = cv::format("%.2f", i.prob);
      label = obj_names[i.obj_id] + ":" + label;

      int baseLine;
      cv::Size labelSize =
          getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      int top = std::max((int)y, labelSize.height);

      cv::rectangle(mat_img, cv::Point(x, y - round(1.5 * labelSize.height)),
                    cv::Point(x + round(1.5 * labelSize.width), y + baseLine),
                    cv::Scalar(255, 255, 255), cv::FILLED);
      putText(mat_img, label, cv::Point2f(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.75,
              cv::Scalar(0, 0, 0), 1);
    }
  }
}

void show_console_result(std::vector<bbox_t> const result_vec,
                         std::vector<std::string> const obj_names,
                         int frame_id = -1) {
  if (frame_id >= 0)
    std::cout << " Frame: " << frame_id << std::endl;
  for (auto &i : result_vec) {
    if (obj_names.size() > i.obj_id)
      std::cout << obj_names[i.obj_id];
    printf(" %.3f | ", i.prob);
  }
  std::cout << std::endl;
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
  std::ifstream file(filename);
  std::vector<std::string> file_lines;
  if (!file.is_open())
    return file_lines;
}

int main(int argc, char *argv[]) {
  std::cout << cv::getBuildInformation() << std::endl;
  std::string names_file = "darknet/data/coco.names";
  auto obj_names = objects_names_from_file(names_file);

  if (argc < 4) {
    perror("Usage: ./detection darknet/cfg/yolov3_x_x.cfg resize_height "
           "resize_width path_to_img_folder\n");
    return 1;
  }

  std::string weights_file = "darknet/yolov4.weights";
  std::string cfg_file = argv[1];
  int height = atoi(argv[2]);
  int width = atoi(argv[3]);

  Detector *detector = new Detector(cfg_file, weights_file);
  Mat image;
  Mat image2;
  std::vector<bbox_t> result_vec;

  // DIR* dirp = opendir(argv[4]);
  // struct dirent * dp;

  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> spent;
  // int i = 0;

  // while ((dp = readdir(dirp)) != NULL) {
  // if( strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0 ) {
  // std::string name = dp->d_name;
  // image = cv::imread(argv[4]);  // argv[4]+name, 1);
  // std::cout << image.shape;
  // cv::imshow(image);
  image = imread(argv[4]);
  // image2 = Mat::zeros(Size(height,width),CV_8UC1);
  std::cout << "\n\n" << image.size() << "\n" << image2.size() << "\n\n";
  if (image.empty()) {
    printf("image not loaded \n");
    return -1;
  }
  // std::cout << image.cols << ":" << image.rows;
  cv::resize(image, image2, Size(height, width), 0, 0, cv::INTER_NEAREST);
  std::cout << "\n\n" << image2.size() << "\n\n";
  start = std::chrono::steady_clock::now();
  result_vec = detector->detect(image2);

  end = std::chrono::steady_clock::now();
  spent = end - start;
  std::cout << spent.count() << " | ";
  show_console_result(result_vec, obj_names);
  ///////////

  start = std::chrono::steady_clock::now();
  detector_gpu_t &detector_gpu =
      *static_cast<detector_gpu_t *>(detector->detector_gpu_ptr.get());
  network &net = detector_gpu.net;
  resize_network(&net, 320, 320);
  end = std::chrono::steady_clock::now();
  spent = end - start;
  std::cout << spent.count() << "\n\n";
  /////////
  cv::resize(image, image2, cv::Size(320, 320), 0, 0, cv::INTER_NEAREST);
  start = std::chrono::steady_clock::now();
  result_vec = detector->detect(image2);

  end = std::chrono::steady_clock::now();
  spent = end - start;
  std::cout << spent.count() << " | ";
  show_console_result(result_vec, obj_names);

  // start = std::chrono::steady_clock::now();
  // delete detector;
  // Detector* detector2 = new Detector(cfg_file, weights_file);
  // end = std::chrono::steady_clock::now();
  // spent = end - start;
  // std::cout << spent.count() << "\n\n";
  // i++;
  //	}
  //	}

  //  closedir(dirp);

  // draw_boxes(image2, result_vec, obj_names, 1);
  // cv::imshow("Result", image2);
  // cv::waitKey(0);
}
