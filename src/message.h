#ifndef __MESSAGE_H__
#define __MESSAGE_H__

#include "env_time.h"
#include "tcp.h"
#include "yolo_v2_class.hpp"
#include <chrono>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <string>

#define MSG_TYPE_FRAME 0x00000001
#define MSG_TYPE_DETECTION 0x00000002
#define MSG_TYPE_STOP 0x00000004

/**
 * @brief Application message on the network follows the following structure.
 *  ______________________________
 * | MsgHdr | Variable length Msg |
 * |________|_____________________|
 *
 */
typedef struct MessageHeader {
  int msgType;
  int msgLength;
  uint64 sendTime;
  uint64 recvTime;
} MessageHeader_t;

typedef struct Message {
  MessageHeader_t msgHdr;
  std::vector<unsigned char> msgData;
} Message_t;

class MessageHandler {

public:
  MessageHandler(TcpConnection &tcp_handle) : tcpHandle_m(tcp_handle) {}
  Message_t recvMsg() const {
    Message_t msg;
    // Read message header
    tcpHandle_m.recv((unsigned char *)(&(msg.msgHdr)),
                     (size_t)sizeof(msg.msgHdr));

    // validate received message
    bool ok = validMsg(msg);
    if (!ok) {
      throw std::runtime_error("Invalid message");
    }

    // Read message data
    msg.msgData.resize(msg.msgHdr.msgLength);
    tcpHandle_m.recv((unsigned char *)msg.msgData.data(),
                     (size_t)msg.msgHdr.msgLength);

    msg.msgHdr.recvTime = EnvTime::Default()->NowMicros();
    return msg;
  }

  int sendMsg(Message_t &msg) const {

    msg.msgHdr.sendTime = EnvTime::Default()->NowMicros();

    // Write message header
    tcpHandle_m.send((unsigned char *)&(msg.msgHdr),
                     (size_t)sizeof(msg.msgHdr));

    // Write message data
    tcpHandle_m.send((unsigned char *)msg.msgData.data(),
                     (size_t)msg.msgHdr.msgLength);

    return 0;
  }

private:
  TcpConnection &tcpHandle_m;

  bool validMsg(Message_t &msg) const {
    if (!((msg.msgHdr.msgType & MSG_TYPE_FRAME) ||
          (msg.msgHdr.msgType & MSG_TYPE_DETECTION))) {
      return false;
    }
    return true;
  }
};

typedef struct FrameHeader {
  int frameId;
  uint64 captureTime;

  /* Timestamps. TODO: some of the fields could be redundant because, msg header
   also carries some of these fields. */
  uint64 clientSendTime;
  uint64 clientRecvTime;
  uint64 serverSendTime;
  uint64 serverRecvTime;
  unsigned int correctModel;
  unsigned int cocoID;
  unsigned int orgWidth;
  unsigned int orgHeight;
  unsigned int serializedSize; // Network length including this header.
} FrameHeader_t;

class FrameObject {
public:
  /* default constructor */
  FrameObject() {}
  /*delete copy constructor and assignment. As this object is going to move
   * around from one thread to another */
  FrameObject(const FrameObject &) = delete;
  FrameObject &operator=(const FrameObject &) = delete;

  FrameHeader_t frameHdr;
  cv::Mat frameMat;
  std::vector<int> compression_params = {cv::IMWRITE_JPEG_QUALITY, 90};

  int resize(int width, int height) {
    // cv::cvtColor(frameMat, frameMat, cv::COLOR_BGR2GRAY);
    cv::resize(frameMat, frameMat, cv::Size(width, height), 1, 1,
               cv::INTER_NEAREST);
  }

  static std::shared_ptr<FrameObject> constructFromMsg(Message_t &msg) {
    assert(msg.msgHdr.msgType & MSG_TYPE_FRAME);
    auto frameMsg = std::make_shared<FrameObject>();
    // frameMsg.msgData = std::move(msg.msgData);
    unsigned char *data_ptr = msg.msgData.data();

    // deserialize frame header
    frameMsg->frameHdr = *(FrameHeader_t *)data_ptr;
    data_ptr += sizeof(FrameHeader_t);

    // deserialize cv frame
    unsigned int frameLen = msg.msgHdr.msgLength - sizeof(FrameHeader_t);
    std::vector<unsigned char> temp_vec(data_ptr, data_ptr + frameLen);
    frameMsg->frameMat = cv::imdecode(temp_vec, 1);

    if (frameMsg->frameMat.empty()) {
      return nullptr;
    }

    frameMsg->frameHdr.serializedSize = msg.msgHdr.msgLength;
    return frameMsg;
  }

  Message_t constructToMsg() {
    Message_t msg;
    msg.msgHdr.msgType = MSG_TYPE_FRAME;
    auto &msgData = msg.msgData;

    msgData.resize(sizeof(FrameHeader_t));
    std::memcpy(msgData.data(), &frameHdr, sizeof(FrameHeader_t));

    // encode frame
    std::vector<unsigned char> encodedVec;
    cv::imencode(".jpg", frameMat, encodedVec, compression_params);
    // std::cout << "STATS:EncodeSize,FrameID:" << frameHdr.frameId << ",Value:"
    // << encodedVec.size() << std::endl;

    // copy encoded data
    msg.msgHdr.msgLength = msgData.size() + encodedVec.size();
    msgData.resize(msg.msgHdr.msgLength);
    std::move(encodedVec.begin(), encodedVec.end(),
              msgData.begin() + sizeof(FrameHeader_t));

    frameHdr.serializedSize = msg.msgHdr.msgLength;
    return msg;
  }
};

typedef struct DetectionHeader {
  FrameHeader_t frameHdr;
  unsigned int usedModel;
  unsigned int totalDetection;
  unsigned int serializedSize; // Network length including this header
} DetectionHeader_t;

typedef struct bbox_t DetectionObject_t;

class DetectionObject {
public:
  /* default constructor */
  DetectionObject() {}
  /* delete copy constructor and assignment. As this object is going to move
   * around from thread to thread. */
  DetectionObject(const DetectionObject &) = delete;
  DetectionObject &operator=(const DetectionObject &) = delete;

  DetectionHeader_t detectionHdr;
  std::vector<DetectionObject_t> detectionObjects;

  static std::shared_ptr<DetectionObject> construct(Message_t &msg) {
    assert(msg.msgHdr.msgType & MSG_TYPE_DETECTION);
    auto detectionMsg = std::make_shared<DetectionObject>();
    unsigned char *data_ptr = msg.msgData.data();

    // deserialize object detection header
    detectionMsg->detectionHdr = *(DetectionHeader_t *)data_ptr;
    data_ptr += sizeof(DetectionHeader_t);

    auto &detectionObjects = detectionMsg->detectionObjects;
    detectionObjects.resize(detectionMsg->detectionHdr.totalDetection);
    std::memcpy(detectionObjects.data(), (DetectionObject_t *)data_ptr,
                detectionMsg->detectionHdr.totalDetection *
                    sizeof(DetectionObject_t));

    detectionMsg->detectionHdr.serializedSize = msg.msgHdr.msgLength;
    return detectionMsg;
  }

  Message_t constructToMsg() {
    Message_t msg;
    msg.msgHdr.msgType = MSG_TYPE_DETECTION;
    auto &msgData = msg.msgData;
    msgData.resize(sizeof(DetectionHeader_t));
    unsigned char *data_ptr = msg.msgData.data();
    std::memcpy(data_ptr, &detectionHdr, sizeof(DetectionHeader_t));

    // copy detected objects. TODO: Check if copy can be prevented.
    msg.msgHdr.msgLength =
        msgData.size() + (detectionObjects.size() * sizeof(DetectionObject_t));
    msgData.resize(msg.msgHdr.msgLength);
    data_ptr = msg.msgData.data() + sizeof(DetectionHeader_t);
    std::memcpy((DetectionObject_t *)data_ptr, detectionObjects.data(),
                detectionObjects.size() * sizeof(DetectionObject_t));

    detectionHdr.serializedSize = msg.msgHdr.msgLength;
    return msg;
  }
};
#endif /* __MESSAGE_H__ */
