#ifndef __FRAME_READER_H__
#define __FRAME_READER_H__
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;

class FrameReader {
public:
  virtual bool next(cv::Mat &) = 0;
  static std::shared_ptr<FrameReader> createInstance(std::string, std::string);
};

class ImageReader : public FrameReader {
public:
  ImageReader(std::string imageList) : inputStream_m(imageList) {}

  bool next(cv::Mat &frame) {
    std::string imgpath;

    if (getline(inputStream_m, imgpath)) {
      frame = imread(imgpath);
      return true;
    }
    return false;
  }

private:
  std::ifstream inputStream_m;
};

class VideoReader : public FrameReader {
public:
  VideoReader(std::string video_path) {
    capture_m.open(video_path, CAP_FFMPEG);
    assert(capture_m.isOpened() && "Cannot open video path");
  }

  bool next(cv::Mat &frame) {
    capture_m.read(frame);
    if (frame.empty()) {
      return false;
    }
    return true;
  }

private:
  VideoCapture capture_m;
};

std::shared_ptr<FrameReader>
FrameReader::createInstance(std::string videoPath, std::string imageList) {
  std::shared_ptr<FrameReader> frameReader;
  if (videoPath != "") {
    frameReader.reset(new VideoReader(videoPath));
  } else {
    frameReader.reset(new ImageReader(imageList));
  }
  return frameReader;
}

#endif /* __FRAME_READER_H__ */