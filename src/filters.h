#ifndef __FILTERS_H__
#define __FILTERS_H__

class SmoothingFilter {
public:
  virtual void update(double) = 0;
  virtual double predict() const = 0;
  virtual void reset(){};
};

class EWMAFilter : public SmoothingFilter {
public:
  EWMAFilter(double weight = 0.5) : weight_m(weight) { reset(); }
  void reset() override { prev_value_m = -1; }
  void update(double value) override {
    if (prev_value_m == -1) {
      prev_value_m = value;
    } else {
      prev_value_m = weight_m * prev_value_m + (1.0 - weight_m) * value;
    }
  }

  double predict() const override { return prev_value_m; }

private:
  double prev_value_m;
  double weight_m = 0.5;
};

/**
 * @brief Adaptive error-based filter proposed by, Kim et al. in "Mobile Network
 * Estimation"
 *
 */
class ErrorBasedFilter : public SmoothingFilter {
#define MAX_HISTORY 10
public:
  ErrorBasedFilter(double gamma = 0.6) : gamma_m(gamma) {
    err_history_m.resize(MAX_HISTORY);
    reset();
  }

  void reset() override {
    prev_est_m = -1;
    prev_err_est_m = -1;
    err_idx_m = 0;
    std::fill(err_history_m.begin(), err_history_m.end(), 0.0);
  }

  void update(double O_t) override {
    if (prev_est_m == -1) {
      prev_est_m = O_t;
    } else {
      // update error using EWMA
      update_err(O_t);

      // compute weight alpha
      double err_max =
          *std::max_element(err_history_m.begin(), err_history_m.end());
      double alpha = 1 - (prev_err_est_m / err_max);

      // estimate value
      prev_est_m = alpha * prev_est_m + (1.0 - alpha) * O_t;
    }
  }

  double predict() const override { return prev_est_m; }

private:
  void update_err(double O_t) {
    double err = abs(prev_est_m - O_t);
    // estimate err
    if (prev_err_est_m == -1) {
      prev_err_est_m = err;
    } else {
      prev_err_est_m = gamma_m * prev_err_est_m + (1.0 - gamma_m) * err;
    }
    err_history_m[err_idx_m++ % MAX_HISTORY] = prev_err_est_m;
  }
  std::vector<double> err_history_m;
  int err_idx_m = 0;
  double prev_err_est_m = -1;
  double gamma_m;
  double prev_est_m = -1;
};

#endif /* __FILTERS_H__ */