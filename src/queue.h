#ifndef __QUEUE__H__
#define __QUEUE_H__

#include <condition_variable>
#include <vector>

#define MAX_QUEUE_SIZE 1024

/**
 * @brief Threadsafe Queue.
 * @todo can we make it lock-less?
 * @tparam DType
 */
template <typename DType> class Queue {
public:
  Queue() : max_size_(MAX_QUEUE_SIZE) {}
  Queue(const int max_size) : max_size_(max_size) {}
  bool enqueue(const DType &item);
  bool dequeue(DType &item, unsigned long timeout);
  bool dequeue(std::vector<DType> &item, int size);
  bool front(DType &item);

private:
  std::queue<DType> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  int max_size_;
};

template <typename DType> bool Queue<DType>::enqueue(const DType &item) {
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.push(item);
  lock.unlock();
  cond_.notify_one();
  return true;
}

template <typename DType>
bool Queue<DType>::dequeue(DType &item, unsigned long timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (queue_.empty()) {
    if (!timeout) {
      cond_.wait(lock);
    } else if (cond_.wait_for(lock, std::chrono::milliseconds(timeout)) ==
               std::cv_status::timeout) {
      return false;
    }
  }
  item = queue_.front();
  queue_.pop();
  return true;
}

template <typename DType> bool Queue<DType>::front(DType &item) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (queue_.empty()) {
    cond_.wait(lock);
  }
  item = queue_.front();
  return true;
}

/**
 * @brief Retrieve batch of elements in one go.
 *        A client of this Queue may starve when the Queue has other clients
 *        that retrieves lesser number of elements.
 *
 * @tparam DType
 * @param item Vector of elements to be returned
 * @param size Number of elements to be returned
 * @return true
 * @return false
 */
template <typename DType>
bool Queue<DType>::dequeue(std::vector<DType> &items, int size) {
  std::unique_lock<std::mutex> lock(mutex_);
  while (queue_.empty() || queue_.size < size) {
    cond_.wait(lock);
  }

  // @todo Need a range iterator which std::Queue doesn't support.
  // use std::deque.
  for (int i = 0; i < size; ++i) {
    // @todo: can we use std::move?
    DType item = queue_.front();
    items.insert(item);
    queue_.pop();
  }

  return true;
}

#endif /* __QUEUE_H__ */
