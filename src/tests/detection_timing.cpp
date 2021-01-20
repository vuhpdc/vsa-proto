#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "yolo_v2_class.hpp" // imported functions from DLL

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "common.h"
#include "env_time.h"
#include "image.h"

#include <dirent.h>
#include <sys/types.h>

static std::unique_ptr<Detector> detector;
static std::unique_ptr<Detector> detector2;

Detector *detectors[2];

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
  for (std::string line; getline(file, line);)
    file_lines.push_back(line);
  std::cout << "object names loaded \n";
  return file_lines;
}

static void print_cocos(FILE *fp, int image_id, std::vector<bbox_t> result_vec,
                        int w, int h, int org_w, int org_h) {
  for (auto &i : result_vec) {
    // std::cout << w << " ," << h << " ," << org_w << " ," << org_h <<
    // std::endl; std::cout << i.x << " ," << i.y << " ," << i.w << " ," << i.h
    // << " ," << std::endl;
    float bx = (i.x / (w * 1.0 / org_w * 1.0));
    float bw = (i.w / (w * 1.0 / org_w * 1.0));
    float by = (i.y / (h * 1.0 / org_h * 1.0));
    float bh = (i.h / (h * 1.0 / org_h * 1.0));

    // std::cout << bx << " ," << by << " ," << bw << " ," << bh << " ," <<
    // std::endl << std::endl;

    char buff[1024];
    sprintf(buff,
            "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], "
            "\"score\":%f},\n",
            image_id, coco_ids[i.obj_id], bx, by, bw, bh, i.prob);
    fprintf(fp, buff);
  }
}

char *copy_string(char *s) {
  if (!s) {
    return NULL;
  }
  char *copy = (char *)malloc(strlen(s) + 1);
  strncpy(copy, s, strlen(s) + 1);
  return copy;
}

char *basecfg(char *cfgfile) {
  char *c = cfgfile;
  char *next;
  while ((next = strchr(c, '/'))) {
    c = next + 1;
  }
  if (!next)
    while ((next = strchr(c, '\\'))) {
      c = next + 1;
    }
  c = copy_string(c);
  next = strchr(c, '.');
  if (next)
    *next = 0;
  return c;
}

int main(int argc, char *argv[]) {
  std::string names_file = "darknet/data/coco.names";
  auto obj_names = objects_names_from_file(names_file);

  if (argc < 4) {
    perror("Usage: ./detection darknet/cfg/yolov3_x_x.cfg resize_height \
              resize_width encode quality (1 to 100) path_to_img_folder_txt\n");
    return 1;
  }

  std::string weights_file = "darknet/yolov4.weights";
  std::string cfg_file = argv[1];
  int height = atoi(argv[2]);
  int width = atoi(argv[3]);
  int org_height, org_width, coco_im_id;
  int quality = atoi(argv[4]);

  char buff[1024];
  FILE *fp = 0;
  snprintf(buff, 1024, "coco_results_%d_%d.json", width, quality);
  fp = fopen(buff, "w");
  fprintf(fp, "[\n");

  // std::vector<Detector> detectors;
  // Detector* detectors[2] = {new Detector(cfg_file1, weights_file1), new
  // Detector(cfg_file1, weights_file1)};

  detectors[0] = new Detector(cfg_file, weights_file);
  std::vector<bbox_t> result_vec;
  std::vector<unsigned char> encodedVec;
  /*  std::vector<unsigned char> encodedVec_jpg10;
    std::vector<unsigned char> encodedVec_jpg20;
    std::vector<unsigned char> encodedVec_jpg30;
    std::vector<unsigned char> encodedVec_jpg40;
    std::vector<unsigned char> encodedVec_jpg50;
    std::vector<unsigned char> encodedVec_jpg60;
    std::vector<unsigned char> encodedVec_jpg70;
    std::vector<unsigned char> encodedVec_jpg80;
    std::vector<unsigned char> encodedVec_jpg90;
    std::vector<unsigned char> encodedVec_jpg100;

    std::vector<unsigned char> encodedVec_png1;
    std::vector<unsigned char> encodedVec_png3;
    std::vector<unsigned char> encodedVec_png6;
    std::vector<unsigned char> encodedVec_png9; */

  cv::Mat readImage, resizedImage, decodedImage;
  std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, quality};
  /*  std::vector<int> compression_params_10 = {cv::IMWRITE_JPEG_QUALITY, 10};
    std::vector<int> compression_params_20 = {cv::IMWRITE_JPEG_QUALITY, 20};
    std::vector<int> compression_params_30 = {cv::IMWRITE_JPEG_QUALITY, 30};
    std::vector<int> compression_params_40 = {cv::IMWRITE_JPEG_QUALITY, 40};
    std::vector<int> compression_params_50 = {cv::IMWRITE_JPEG_QUALITY, 50};
    std::vector<int> compression_params_60 = {cv::IMWRITE_JPEG_QUALITY, 60};
    std::vector<int> compression_params_70 = {cv::IMWRITE_JPEG_QUALITY, 70};
    std::vector<int> compression_params_80 = {cv::IMWRITE_JPEG_QUALITY, 80};
    std::vector<int> compression_params_90 = {cv::IMWRITE_JPEG_QUALITY, 90};
    std::vector<int> compression_params_100 = {cv::IMWRITE_JPEG_QUALITY, 100};

    std::vector<int> compression_params_1 = {cv::IMWRITE_PNG_COMPRESSION, 1};
    std::vector<int> compression_params_3 = {cv::IMWRITE_PNG_COMPRESSION, 3};
    std::vector<int> compression_params_6 = {cv::IMWRITE_PNG_COMPRESSION, 6};
    std::vector<int> compression_params_9 = {cv::IMWRITE_PNG_COMPRESSION, 9};

          std::cout<< "id, imread, startsize, resize, resizedsize, "
                        << "jpg_10_en, jpg_10_de, ensize_10, "
                        << "jpg_20_en, jpg_20_de, ensize_20, "
                        << "jpg_30_en, jpg_30_de, ensize_30, "
                        << "jpg_40_en, jpg_40_de, ensize_40, "
                        << "jpg_50_en, jpg_50_de, ensize_50, "
                        << "jpg_60_en, jpg_60_de, ensize_60, "
                        << "jpg_70_en, jpg_70_de, ensize_70, "
                        << "jpg_80_en, jpg_80_de, ensize_80, "
                        << "jpg_90_en, jpg_90_de, ensize_90, "
                        << "jpg_100_en, jpg_100_de, ensize_100, "
                        << "png_1_en, png_1_de, ensize_1, "
                        << "png_3_en, png_3_de, ensize_3, "
                        << "png_6_en, png_6_de, ensize_6, "
                        << "png_9_en, png_9_de, ensize_9" << std::endl; */

  int i = 0;
  // DIR* dirp = opendir(argv[5]);
  // struct dirent * dp;
  std::ifstream input(argv[5]);

  // while ((dp = readdir(dirp)) != NULL && i < 1000) {
  // if( strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0 ) {
  // std::string name = dp->d_name;
  // char imgpath[1024];
  // snprintf(imgpath, 1024, "%s%s", argv[5], name.c_str());

  for (std::string imgpath; getline(input, imgpath);) {
    std::cout << imgpath << std::endl;
    auto startTime = EnvTime::Default()->NowMicros();
    readImage = cv::imread(imgpath);
    if (!readImage.data) {
      printf("No image data \n");
      return -1;
    }

    coco_im_id = atoi(basecfg(&imgpath[0]));
    org_height = readImage.rows;
    org_width = readImage.cols;

    auto timepoint1 = EnvTime::Default()->NowMicros();
    cv::resize(readImage, resizedImage, cv::Size(width, height), 1, 1,
               cv::INTER_NEAREST);
    auto timepoint2 = EnvTime::Default()->NowMicros();
    cv::imencode(".jpg", resizedImage, encodedVec, compression_params);
    auto timepoint3 = EnvTime::Default()->NowMicros();
    decodedImage = cv::imdecode(encodedVec, 1);
    auto timepoint4 = EnvTime::Default()->NowMicros();
    result_vec = detectors[0]->detect(decodedImage);
    auto timepoint5 = EnvTime::Default()->NowMicros();
    print_cocos(fp, coco_im_id, result_vec, width, height, org_width,
                org_height);
    // show_console_result(result_vec, obj_names);
    auto timepoint6 = EnvTime::Default()->NowMicros();

    double time1 = TIME_US_TO_MS(timepoint1 - startTime);
    double time2 = TIME_US_TO_MS(timepoint2 - timepoint1);
    double time3 = TIME_US_TO_MS(timepoint3 - timepoint2);
    double time4 = TIME_US_TO_MS(timepoint4 - timepoint3);
    double time5 = TIME_US_TO_MS(timepoint5 - timepoint4);
    double time6 = TIME_US_TO_MS(timepoint6 - timepoint5);

    std::cout << i << ", " << time1 << ", " << time2 << ", " << time3 << ", "
              << time4 << ", " << time5 << ", " << time6 << std::endl;

    /*    cv::imencode(".jpg", resizedImage, encodedVec_jpg10,
       compression_params_10); auto timepoint3 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg10, 1);
          //imwrite("jpg10.jpg", decodedImage);
          auto timepoint4 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImage, encodedVec_jpg20,
       compression_params_20); auto timepoint5 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg20, 1);
          //imwrite("jpg20.jpg", decodedImage);
          auto timepoint6 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImage, encodedVec_jpg30,
       compression_params_30); auto timepoint7 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg30, 1);
          //imwrite("jpg30.jpg", decodedImage);
          auto timepoint8 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImage, encodedVec_jpg40,
       compression_params_40); auto timepoint9 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg40, 1);
          //imwrite("jpg40.jpg", decodedImage);
          auto timepoint10 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImagfprintf(fp, "\n]\n");
            fclose(fp);e, encodedVec_jpg50, compression_params_50);
          auto timepoint11 = EnvTime::Default()->NowMicros();
          decodedImage = cv::imdecode(encodedVec_jpg50, 1);
          //imwrite("jpg50.jpg", decodedImage);
          auto timepoint12 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImage, encodedVec_jpg60,
       compression_params_60); auto timepoint13 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg60, 1);
          //imwrite("jpg60.jpg", decodedImage);
          auto timepoint14 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImage, encodedVec_jpg70,
       compression_params_70); auto timepoint15 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg70, 1);
          //imwrite("jpg70.jpg", decodedImage);
          auto timepoint16 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImage, encodedVec_jpg80,
       compression_params_80); auto timepoint17 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg80, 1);
          //imwrite("jpg80.jpg", decodedImage);
          auto timepoint18 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImage, encodedVec_jpg90,
       compression_params_90); auto timepoint19 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg90, 1);
          //imwrite("jpg90.jpg", decodedImage);
          auto timepoint20 = EnvTime::Default()->NowMicros();

          cv::imencode(".jpg", resizedImage, encodedVec_jpg100,
       compression_params_100); auto timepoint21 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_jpg100, 1);
          //imwrite("jpg100.jpg", decodedImage);
          auto timepoint22 = EnvTime::Default()->NowMicros();


          cv::imencode(".png", resizedImage, encodedVec_png1,
       compression_params_1); auto timepoint23 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_png1, 1);
          //imwrite("png1.jpg", decodedImage);
          auto timepoint24 = EnvTime::Default()->NowMicros();

          cv::imencode(".png", resizedImage, encodedVec_png3,
       compression_params_3); auto timepoint25 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_png3, 1);
          //imwrite("png3.jpg", decodedImage);
          auto timepoint26 = EnvTime::Default()->NowMicros();

          cv::imencode(".png", resizedImage, encodedVec_png6,
       compression_params_6); auto timepoint27 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_png6, 1);
          //imwrite("png6.jpg", decodedImage);
          auto timepoint28 = EnvTime::Default()->NowMicros();

          cv::imencode(".png", resizedImage, encodedVec_png9,
       compression_params_9); auto timepoint29 =
       EnvTime::Default()->NowMicros(); decodedImage =
       cv::imdecode(encodedVec_png9, 1);
          //imwrite("png9.jpg", decodedImage);
          auto timepoint30 = EnvTime::Default()->NowMicros();





          //cv::imshow("decoded", decodedImage);
                            //cv::waitKey(0);

          double time1 = TIME_US_TO_MS(timepoint1 - startTime);
          double time2 = TIME_US_TO_MS(timepoint2 - timepoint1);
          double time3 = TIME_US_TO_MS(timepoint3 - timepoint2);
          double time4 = TIME_US_TO_MS(timepoint4 - timepoint3);
          double time5 = TIME_US_TO_MS(timepoint5 - timepoint4);
          double time6 = TIME_US_TO_MS(timepoint6 - timepoint5);
          double time7 = TIME_US_TO_MS(timepoint7 - timepoint6);
          double time8 = TIME_US_TO_MS(timepoint8 - timepoint7);
          double time9 = TIME_US_TO_MS(timepoint9 - timepoint8);
          double time10 = TIME_US_TO_MS(timepoint10 - timepoint9);
          double time11 = TIME_US_TO_MS(timepoint11 - timepoint10);
          double time12 = TIME_US_TO_MS(timepoint12 - timepoint11);
          double time13 = TIME_US_TO_MS(timepoint13 - timepoint12);
          double time14 = TIME_US_TO_MS(timepoint14 - timepoint13);
          double time15 = TIME_US_TO_MS(timepoint15 - timepoint14);
          double time16 = TIME_US_TO_MS(timepoint16 - timepoint15);
          double time17 = TIME_US_TO_MS(timepoint17 - timepoint16);
          double time18 = TIME_US_TO_MS(timepoint18 - timepoint17);
          double time19 = TIME_US_TO_MS(timepoint19 - timepoint18);
          double time20 = TIME_US_TO_MS(timepoint20 - timepoint19);
          double time21 = TIME_US_TO_MS(timepoint21 - timepoint20);
          double time22 = TIME_US_TO_MS(timepoint22 - timepoint21);
          double time23 = TIME_US_TO_MS(timepoint23 - timepoint22);
          double time24 = TIME_US_TO_MS(timepoint24 - timepoint23);
          double time25 = TIME_US_TO_MS(timepoint25 - timepoint24);
          double time26 = TIME_US_TO_MS(timepoint26 - timepoint25);
          double time27 = TIME_US_TO_MS(timepoint27 - timepoint26);
          double time28 = TIME_US_TO_MS(timepoint28 - timepoint27);
          double time29 = TIME_US_TO_MS(timepoint29 - timepoint28);
          double time30 = TIME_US_TO_MS(timepoint30 - timepoint29);


          auto startsize = readImage.total() * readImage.elemSize();
          auto resizedsize = resizedImage.total() * resizedImage.elemSize();
          auto ensize_10 = encodedVec_jpg10.size();
          auto ensize_20 = encodedVec_jpg20.size();
          auto ensize_30 = encodedVec_jpg30.size();
          auto ensize_40 = encodedVec_jpg40.size();
          auto ensize_50 = encodedVec_jpg50.size();
          auto ensize_60 = encodedVec_jpg60.size();
          auto ensize_70 = encodedVec_jpg70.size();
          auto ensize_80 = encodedVec_jpg80.size();
          auto ensize_90 = encodedVec_jpg90.size();
          auto ensize_100 = encodedVec_jpg100.size();
          auto ensize_1 = encodedVec_png1.size();
          auto ensize_3 = encodedVec_png3.size();
          auto ensize_6 = encodedVec_png6.size();
          auto ensize_9 = encodedVec_png9.size();

          std::cout <<  i << ", " << time1 << ", " << startsize << ", "
              << time2 << ", " << resizedsize << ", "
              << time3 << ", " << time4 << ", " << ensize_10 << ", "
              << time5 << ", " << time6 << ", " << ensize_20 << ", "
              << time7 << ", " << time8 << ", " << ensize_30 << ", "
              << time9 << ", " << time10 << ", " << ensize_40 << ", "
              << time11 << ", " << time12 << ", " << ensize_50 << ", "
              << time13 << ", " << time14 << ", " << ensize_60 << ", "
              << time15 << ", " << time16 << ", " << ensize_70 << ", "
              << time17 << ", " << time18 << ", " << ensize_80 << ", "
              << time19 << ", " << time20 << ", " << ensize_90 << ", "
              << time21 << ", " << time22 << ", " << ensize_100 << ", "
              << time23 << ", " << time24 << ", " << ensize_1 << ", "
              << time25 << ", " << time26 << ", " << ensize_3 << ", "
              << time27 << ", " << time28 << ", " << ensize_6 << ", "
              << time29 << ", " << time30 << ", " << ensize_9 << std::endl;
    */
    i++;
    //}
  }

  // closedir(dirp);

  fseek(fp, -2, SEEK_CUR);
  fprintf(fp, "\n]\n");
  fclose(fp);

  // draw_boxes(image2, result_vec, obj_names, 1);
  // cv::imshow("Result", image2);
  // cv::waitKey(0);
}
