/**
 * The code is borrowed from tensorflow codebase.
 * ref: tensorflow/core/platform/env_time.h
 */

#ifndef __ENV_TIME_H__
#define __ENV_TIME_H__

#include <sys/time.h>
#include <time.h>

#define TIME_US_TO_MS(time) (double(time) / 1000.0)

// typedef unsigned long long uint64;

/// \brief An interface used by the tensorflow implementation to
/// access timer related operations.
class EnvTime {
public:
  static constexpr uint64 kMicrosToPicos = 1000ULL * 1000ULL;
  static constexpr uint64 kMicrosToNanos = 1000ULL;
  static constexpr uint64 kMillisToMicros = 1000ULL;
  static constexpr uint64 kMillisToNanos = 1000ULL * 1000ULL;
  static constexpr uint64 kSecondsToMillis = 1000ULL;
  static constexpr uint64 kSecondsToMicros = 1000ULL * 1000ULL;
  static constexpr uint64 kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

  EnvTime();
  virtual ~EnvTime() = default;

  /// \brief Returns a default impl suitable for the current operating
  /// system.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static EnvTime *Default();

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  virtual uint64 NowNanos() const = 0;

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64 NowMicros() const { return NowNanos() / kMicrosToNanos; }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64 NowSeconds() const { return NowNanos() / kSecondsToNanos; }
};

class PosixEnvTime : public EnvTime {
public:
  PosixEnvTime() {}

  uint64 NowNanos() const override {
    struct timespec ts;
    // clock_gettime(CLOCK_MONOTONIC, &ts);
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64>(ts.tv_nsec));
  }
};

// #if defined(PLATFORM_POSIX) || defined(__ANDROID__)
EnvTime::EnvTime() {}
EnvTime *EnvTime::Default() {
  static EnvTime *default_env_time = new PosixEnvTime;
  return default_env_time;
}

#endif /* __ENV_TIME_H__ */