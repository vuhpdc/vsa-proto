#include <arpa/inet.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <numeric>
#include <pthread.h>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "common.h"
#include "frame_reader.h"

#include "env_time.h"
#include "message.h"

#if defined(MPC_CONTROLLER)
#include "mpc_controller.h"
#elif defined(BASIC_CONTROLLER)
#include "basic_controller.h"
#elif defined(PID_CONTROLLER)
#include "pid_controller.h"
#else
#include "fixed_controller.h"
#endif

using namespace cv;
using namespace std;

vector<string> obj_names;

int capture_frame_height;
int capture_frame_width;
unsigned int max_frames = UINT_MAX - 1;

// function to read all object names from a file so the object id of an object
// can be matched with a name
vector<string> objects_names_from_file(string const filename) {
  ifstream file(filename);
  vector<string> file_lines;
  if (!file.is_open())
    return file_lines;
  for (string line; getline(file, line);)
    file_lines.push_back(line);
  // cout << "object names loaded \n";
  return file_lines;
}

// render the bounding boxes of objects stored in the result vec, which is
// returned by the server, in the current frame
void drawBoxes(frame_obj local_frame_obj, vector<result_obj> result_vec,
               unsigned int curr_frame_id) {
  // for each located object
  for (auto &i : result_vec) {
    int x = (int)(i.x / (n_width[local_frame_obj.correct_model] * 1.0 /
                         capture_frame_width * 1.0));
    int w = (int)(i.w / (n_width[local_frame_obj.correct_model] * 1.0 /
                         capture_frame_width * 1.0));
    int y = (int)(i.y / (n_height[local_frame_obj.correct_model] * 1.0 /
                         capture_frame_height * 1.0));
    int h = (int)(i.h / (n_height[local_frame_obj.correct_model] * 1.0 /
                         capture_frame_height * 1.0));

    rectangle(local_frame_obj.frame, Point(x, y), Point(x + w, y + h),
              Scalar(255, 178, 50), 3);
    if (obj_names.size() > i.obj_id) {
      string label = format("%.2f", i.prob);
      label = obj_names[i.obj_id] + ":" + label;

      int baseLine;
      Size labelSize =
          getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      int top = max((int)y, labelSize.height);

      rectangle(local_frame_obj.frame,
                Point(x, top - round(1.5 * labelSize.height)),
                Point(x + round(1.5 * labelSize.width), top + baseLine),
                Scalar(255, 255, 255), FILLED);
      putText(local_frame_obj.frame, label, Point(x, top), FONT_HERSHEY_SIMPLEX,
              0.75, Scalar(0, 0, 0), 1);
    }
  }
  // add latency of the frame on which detection is performed and the difference
  // between that frame and the current frame
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> spent = end - local_frame_obj.start;
  string label = format("Curr: %d | Inf: %d | Time: %f", curr_frame_id,
                        local_frame_obj.frame_id, spent.count());
  int baseLine;
  Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  int top = max(20, labelSize.height);
  rectangle(local_frame_obj.frame, Point(0, 0),
            Point(round(1.5 * labelSize.width), top + baseLine),
            Scalar(255, 255, 255), FILLED);
  putText(local_frame_obj.frame, label, Point(0, top), FONT_HERSHEY_SIMPLEX,
          0.75, Scalar(0, 0, 0), 1);
}

// print console output for a frame instead of rendering and showing an image
void consoleOutput(frame_obj local_frame_obj, vector<result_obj> result_vec,
                   unsigned int curr_frame_id) {
  // print latency and frame difference between current frame and frame on which
  // detection is performed
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> spent = end - local_frame_obj.start;
  printf("------------------------------------------------------------\n");
  printf("Received inference result from frame %d\n", local_frame_obj.frame_id);
  printf("Frame %d was captured %f seconds ago\n", local_frame_obj.frame_id,
         spent.count());
  printf("Currently captured frame %d is %d frames newer\n", curr_frame_id,
         curr_frame_id - local_frame_obj.frame_id);
  printf("A total of %zu objects have been recognized\n", result_vec.size());

  // print names and confidence of located objects
  for (auto &i : result_vec) {
    if (obj_names.size() > i.obj_id) {
      string label = format("%.2f", i.prob);
      label = obj_names[i.obj_id] + ":" + label;
      cout << label << "\n";
    }
  }
  printf("\n");
}

static void print_cocos(FILE *fp, int image_id,
                        std::vector<DetectionObject_t> result_vec, int w, int h,
                        int org_w, int org_h) {
  for (auto &i : result_vec) {
    float bx = (i.x / (w * 1.0 / org_w * 1.0));
    float bw = (i.w / (w * 1.0 / org_w * 1.0));
    float by = (i.y / (h * 1.0 / org_h * 1.0));
    float bh = (i.h / (h * 1.0 / org_h * 1.0));

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

void signal_handler(int) { kill(0, SIGTERM); }

pid_t shape(void *) {
  if (!ConfigManager::Default()->networkShaping) {
    return 0;
  }

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork failed");
    exit(1);
  } else if (pid == 0) {
    std::cout << "Using network shaping: "
              << ConfigManager::Default()->shapingFile << std::endl;
    assert(ConfigManager::Default()->shapingFile != "" &&
           "shaping file path is empty");
    std::cout << endl;
    execl("/bin/bash", "bash", "../simulation/shape.sh",
          ConfigManager::Default()->shapingFile.c_str());
    exit(0);
  } else {
    // cout << " parent continues " << endl;
  }
  return pid;
}

// receive result from server and process result
void *recv_detection(Controller &controller, MessageHandler &msgHandler) {

  // set output file
  char buff[1024];
  snprintf(buff, 1024,
           ConfigManager::Default()->outputFile.c_str()); //,width,quality);
  FILE *fp = 0;
  fp = fopen(buff, "w");
  fprintf(fp, "[\n");

  while (true) {
    auto msg = msgHandler.recvMsg();
    auto detectionObject = DetectionObject::construct(msg);

    // Mark endtime when we receive detection results
    detectionObject->detectionHdr.frameHdr.serverSendTime = msg.msgHdr.sendTime;
    detectionObject->detectionHdr.frameHdr.clientRecvTime = msg.msgHdr.recvTime;

    auto &detectionHdr = detectionObject->detectionHdr;
    auto &frameHdr = detectionObject->detectionHdr.frameHdr;

    if (frameHdr.frameId == -1) {
      break;
    }

    double time_spent =
        TIME_US_TO_MS(frameHdr.clientRecvTime - frameHdr.clientSendTime);
    // @note: It gives time in milliseconds
    double time_since_capture =
        TIME_US_TO_MS(frameHdr.clientRecvTime - frameHdr.captureTime);

    std::cout << "RecvDetection:" << frameHdr.frameId
              << ",CocoID:" << frameHdr.cocoID
              << ",DesiredModel:" << frameHdr.correctModel
              << ",UsedModel:" << detectionHdr.usedModel
              << ",TotalObjects:" << detectionObject->detectionObjects.size()
              << ",TimeSpent:" << time_spent
              << ",TimeSinceCapture:" << time_since_capture << std::endl;

    auto owd_c_to_s =
        TIME_US_TO_MS(frameHdr.serverRecvTime - frameHdr.clientSendTime);
    auto owd_s_to_c =
        TIME_US_TO_MS(frameHdr.clientRecvTime - frameHdr.serverSendTime);

    controller.update(detectionObject);

    print_cocos(fp, frameHdr.cocoID, detectionObject->detectionObjects,
                n_width[frameHdr.correctModel], n_height[frameHdr.correctModel],
                frameHdr.orgWidth, frameHdr.orgHeight);
  }
  std::cout << "recv detection done" << std::endl;
  fseek(fp, -2, SEEK_CUR);
  fprintf(fp, "\n]\n");
  fclose(fp);
}

void send_frame(MessageQueue_t &frameQueue, MessageHandler &msgHandler) {
  while (true) {
    std::shared_ptr<Message_t> msg;
    frameQueue.dequeue(msg, 0);

    assert(msg->msgHdr.msgType & MSG_TYPE_FRAME ||
           msg->msgHdr.msgType & MSG_TYPE_STOP);
    msgHandler.sendMsg(*msg);
    if (msg->msgHdr.msgType & MSG_TYPE_STOP) {
      break;
    }
  }
}

// read frame, preprocess and send it to the server for object detection
void *read_frame(Controller &controller, MessageQueue_t &frameQueue) {
  // auto prevFrameTime = EnvTime::Default()->NowMicros();

  struct timespec t;
  t.tv_sec = 0;
  double interval = FRAME_ARRIVAL_TIME * 1000000;
  int read_delay = 20000000;
  int frame_counter = 0;

  auto frameReader = FrameReader::createInstance(
      ConfigManager::Default()->videoPath, ConfigManager::Default()->imageList);

  while (frame_counter <= max_frames) {
    auto startTime = EnvTime::Default()->NowMicros();
    auto frameObj = std::make_shared<FrameObject>();
    auto &frameHdr = frameObj->frameHdr;

    if (!frameReader->next(frameObj->frameMat)) {
      perror("empty frame");
      break;
    }
    assert(!frameObj->frameMat.isEmpty() && "Frame is empty");

    auto readTime = EnvTime::Default()->NowMicros();
    t.tv_nsec = read_delay - ((readTime - startTime) * 1000);
    nanosleep(&t, NULL);

    frameHdr.captureTime = EnvTime::Default()->NowMicros();
    frameHdr.frameId = frame_counter;
    frameHdr.cocoID = frame_counter;

    frameHdr.orgWidth = frameObj->frameMat.cols;
    frameHdr.orgHeight = frameObj->frameMat.rows;
    frameHdr.correctModel = controller.predict(frameObj);

    frameObj->resize(n_width[frameHdr.correctModel],
                     n_height[frameHdr.correctModel]);

    auto msg = std::make_shared<Message_t>(frameObj->constructToMsg());
    frameQueue.enqueue(msg);

    frame_counter++;

    // ensure frame reading interval
    auto endTime = EnvTime::Default()->NowMicros();
    t.tv_nsec = interval - ((endTime - startTime) * 1000);
    nanosleep(&t, NULL);
  }

  // construct and communication ending message
  auto frameObj = std::make_shared<FrameObject>();
  auto &frameHdr = frameObj->frameHdr;
  frameHdr.frameId = -1;
  frameObj->frameMat = Mat::zeros(Size(64, 64), CV_8UC1);
  auto msg = std::make_shared<Message_t>(frameObj->constructToMsg());
  msg->msgHdr.msgType |= MSG_TYPE_STOP;
  frameQueue.enqueue(msg);
  std::cout << "send thread done \n" << std::endl;
}

void test_magic_number(MessageHandler &msgHandler) {
  Message_t msg;
  // Send magic number request
  int magic_num = 100;
  // Build header
  msg.msgHdr.msgType = MSG_TYPE_FRAME;
  msg.msgHdr.msgLength = sizeof(int);
  // Build variable length data
  msg.msgData.resize(msg.msgHdr.msgLength);
  std::memcpy(msg.msgData.data(), &magic_num, sizeof(magic_num));

  // Receive magic number response
  msgHandler.sendMsg(msg);
  msg = msgHandler.recvMsg();
  printf("MsgClient address %p %p %p\n", &msg, &msg.msgData,
         msg.msgData.data());
  magic_num = *(int *)msg.msgData.data();
  assert(msg.msgHdr.msgType & MSG_TYPE_DETECTION);
  std::cout << "Test Magic number response " << magic_num << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./client <config file> " << std::endl;
    return 1;
  }

  // Parse config json file
  ConfigManager *configManager = ConfigManager::Default();
  configManager->readConfig(argv[1]);

  string names_file = configManager->darknetDataDir + "/coco.names";
  obj_names = objects_names_from_file(names_file);

  // Start connection
  TcpClient client =
      TcpClient(configManager->serverHost, configManager->serverPort, 0);
  client.open();
  MessageHandler msgHandler(client);
  std::cout << "Connection created with server " << configManager->serverHost
            << configManager->serverPort << std::endl;

  test_magic_number(msgHandler);

  // Define controller
  Controller controller(new Controller::Impl(configManager->startingModel));

  // Frame reader and sender from video source
  MessageQueue_t frameQueue;
  std::thread frameReaderThread(read_frame, std::ref(controller),
                                std::ref(frameQueue));
  std::thread frameSenderThread(send_frame, std::ref(frameQueue),
                                std::ref(msgHandler));

  // Object detection receiver
  std::thread detectionReceiverThread(recv_detection, std::ref(controller),
                                      std::ref(msgHandler));

  signal(SIGINT, signal_handler);
  auto shapePid = shape(NULL);
  // Wait for threads to finish
  frameReaderThread.join();
  frameSenderThread.join();
  detectionReceiverThread.join();

  kill(shapePid, SIGTERM);
  std::cout << "All threads done" << std::endl;
  sleep(10); // wait for shaping to finish
  return 0;
}
