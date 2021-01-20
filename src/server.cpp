#include <arpa/inet.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "network.h"
#include "yolo_v2_class.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "network.h"
#include "yolo_v2_class.hpp"

#include "common.h"
#include "env_time.h"
#include "message.h"
// #include "model_manager_2models.h"
#include "model_manager_pool.h"

using namespace cv;
using namespace std;

// function to read all object names from a file so the object id of an object
// can be matched with a name
vector<string> objects_names_from_file(string const filename) {
  ifstream file(filename);
  vector<string> file_lines;
  if (!file.is_open())
    return file_lines;
  for (string line; getline(file, line);)
    file_lines.push_back(line);
  cout << "object names loaded \n";
  return file_lines;
}

// perform object detection on a received frame and send the result vector to
// the client
void *process_frame(MessageQueue_t &msgQueue, DetectionQueue_t &detectionQueue,
                    ModelManager &modelManager) {

  while (true) {
    std::shared_ptr<Message_t> msg;
    msgQueue.dequeue(msg, 0);
    auto frameObject = FrameObject::constructFromMsg(*msg);
    frameObject->frameHdr.serverRecvTime = msg->msgHdr.recvTime;
    frameObject->frameHdr.clientSendTime = msg->msgHdr.sendTime;

    auto detectionObject = std::make_shared<DetectionObject>();
    auto &detectionHdr = detectionObject->detectionHdr;

    // copy frame header
    auto &frameHdr = frameObject->frameHdr;
    detectionHdr.frameHdr = frameHdr;

    // Perform Object detection for this frame
    detectionObject->detectionObjects = modelManager.detect(
        frameObject->frameMat, frameHdr.correctModel, detectionHdr.usedModel);
    detectionHdr.totalDetection = detectionObject->detectionObjects.size();
    detectionQueue.enqueue(detectionObject);
    printf("process_frame %u %u\n", frameHdr.frameId,
           detectionHdr.totalDetection);
    if (frameHdr.frameId == -1) {
      break;
    }
  }
  std::cout << "process_frame done" << std::endl;
}

// receive a frame and store it in a buffer
void *recv_frame(MessageQueue_t &msgQueue, MessageHandler &msgHandler) {
  while (true) {
    // TODO: Hope this message copy to shared_ptr is not expensive.
    auto msg = std::make_shared<Message_t>(msgHandler.recvMsg());
    assert(msg->msgHdr.msgType & MSG_TYPE_FRAME ||
           msg->msgHdr.msgType & MSG_TYPE_STOP);
    msgQueue.enqueue(msg);
    if (msg->msgHdr.msgType & MSG_TYPE_STOP) {
      break;
    }
  }
  std::cout << "recv_frame done" << std::endl;
}

void *send_detection(DetectionQueue_t &detectionQueue,
                     MessageHandler &msgHandler) {
  while (true) {
    std::shared_ptr<DetectionObject> detectionObject;
    detectionQueue.dequeue(detectionObject, 0);
    auto msg = detectionObject->constructToMsg();
    msgHandler.sendMsg(msg);

    auto modelPredictionTime =
        msg.msgHdr.sendTime -
        detectionObject->detectionHdr.frameHdr.serverRecvTime;
    printf("send_detection:,FrameId:%u,ModelPredictionTime:%f,UsedModel:%d,"
           "DesiredModel:%d\n",
           detectionObject->detectionHdr.frameHdr.frameId,
           TIME_US_TO_MS(modelPredictionTime),
           detectionObject->detectionHdr.usedModel,
           detectionObject->detectionHdr.frameHdr.correctModel);

    if (detectionObject->detectionHdr.frameHdr.frameId == -1) {
      break;
    }
  }
  std::cout << "send_detection done" << std::endl;
}

/**
 * @brief Function to test msgHandler.
 *
 * @param msgHandler
 */
void test_magic_num(MessageHandler &msgHandler) {
  // Receive test magic number request
  Message_t msg = msgHandler.recvMsg();
  int magic_num = *(int *)msg.msgData.data();
  assert(msg.msgHdr.msgType & MSG_TYPE_FRAME);
  std::cout << "Test magic number request " << magic_num << std::endl;

  // Send test magic number response
  magic_num = 101;
  msg.msgHdr.msgType = MSG_TYPE_DETECTION;
  msg.msgHdr.msgLength = sizeof(int);
  msg.msgData.resize(msg.msgHdr.msgLength);
  std::memcpy(msg.msgData.data(), &magic_num, sizeof(magic_num));
  msgHandler.sendMsg(msg);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./server <config file> " << std::endl;
    return 1;
  }

  // Parse config json file
  ConfigManager *configManger = ConfigManager::Default();
  configManger->readConfig(argv[1]);

  while (true) {
    // Create model manager
    ModelManager modelManager(configManger->startingModel);

    // open connection to client
    TcpServer server = TcpServer(configManger->serverPort);
    server.open();
    std::cout << "Server created " << std::endl;
    TcpConnection clientConnection = server.accept_connection();
    std::cout << "Client connection accepted" << std::endl;
    MessageHandler msgHandler(clientConnection);

    test_magic_num(msgHandler);

    // create threads
    MessageQueue_t msgQueue;
    DetectionQueue_t detectionQueue;
    std::thread frameReceiverThread(recv_frame, std::ref(msgQueue),
                                    std::ref(msgHandler));
    std::thread frameProcessorThread(process_frame, std::ref(msgQueue),
                                     std::ref(detectionQueue),
                                     std::ref(modelManager));
    std::thread detectionSenderThread(send_detection, std::ref(detectionQueue),
                                      std::ref(msgHandler));

    // wait for threads
    frameReceiverThread.join();
    frameProcessorThread.join();
    detectionSenderThread.join();
    printf("all threads done\n");
  }

  return 0;
}
