#ifndef __MPC_SOLVER_H__
#define __MPC_SOLVER_H__
#include "common.h"
#include <assert.h>
#include <iostream>
#include <vector>

class Solver {
public:
  explicit Solver(unsigned int horizon, unsigned int totalModels,
                  std::vector<double> modelAccuracy,
                  unsigned int defaultInputSize, double weightForAccuracy,
                  double weightForSwitching)
      : horizon_m(horizon), totalModels_m(totalModels),
        prevInputSize(defaultInputSize), weightForAccuracy_m(weightForAccuracy),
        weightForSwitching_m(weightForSwitching) {}

  virtual std::vector<unsigned int>
  solve(const std::vector<double> &, const std::vector<std::vector<double>> &,
        const std::vector<std::vector<size_t>> &, double, double,
        std::vector<double> &) = 0;

  static Solver *createInstance(unsigned int horizon, unsigned int totalModels,
                                std::vector<double> modelAccuracy,
                                unsigned int defaultInputSize,
                                double weightForAccuracy,
                                double weightForSwitching);

protected:
  unsigned int horizon_m;
  unsigned int totalModels_m;
  double prevInputSize;
  double weightForAccuracy_m;
  double weightForSwitching_m;
};

#if defined(CPLEX_SOLVER)
#ifndef IL_STD
#define IL_STD
#endif
#include <ilcp/cp.h>

class CplexSolver : public Solver {
public:
  explicit CplexSolver(unsigned int horizon, unsigned int totalModels,
                       std::vector<double> modelAccuracy,
                       unsigned int defaultInputSize, double weightForAccuracy,
                       double weightForSwitching)
      : Solver(horizon, totalModels, modelAccuracy, defaultInputSize,
               weightForAccuracy, weightForSwitching) {

    init(modelAccuracy);
  }

  ~CplexSolver() { env_m.end(); }
  std::vector<unsigned int>
  solve(const std::vector<double> &throughputHorizon,
        const std::vector<std::vector<double>> &modelTimeHorizon,
        const std::vector<std::vector<size_t>> &netSizeHorizon,
        double lastModelAccuracy) {
    auto start_time = EnvTime::Default()->NowMicros();
    for (unsigned int horizon = 0; horizon < horizon_m; ++horizon) {
      std::cout << "Horizon " << horizon << "\n"
                << "\tThroughput: " << throughputHorizon[horizon] << "\n";
      for (unsigned int model = 0; model < totalModels_m; ++model) {
        rttPredTime_m[horizon][model] =
            getRTT(throughputHorizon[horizon], modelTimeHorizon[horizon][model],
                   netSizeHorizon[horizon][model]);
        std::cout << "RTT: " << rttPredTime_m[horizon][model]
                  << "\tModelTime: " << modelTimeHorizon[horizon][model]
                  << "\tNetSize: " << netSizeHorizon[horizon][model]
                  << std::endl;
      }
    }
    std::cout << "\n";

    // remove previous constraint values and add new contraints.
    // NOTE: if we simply update rttPredTime, the values are not reflected in
    // the model
    // TODO: If this takes too much time, need to revisit it again.
    model_m.remove(constraints_m);
    createConstraints();
    model_m.add(constraints_m);

    bool solved = false;
    std::vector<unsigned int> inputSize;
    start_time = EnvTime::Default()->NowMicros();
    try {
      solved = cplex_m.solve();
      if (solved) {
        for (unsigned int horizon = 0; horizon < horizon_m; ++horizon) {
          inputSize.push_back(cplex_m.getValue(inputSize_m[horizon]));
        }
      }
    } catch (const IloException &e) {
      std::cerr << "\n\nCPLEX Raised an exception:\n";
      std::cerr << e << "\n";
    } catch (...) {
      std::cerr << "The following unknown exception was found: " << std::endl;
    }
    auto end_time = EnvTime::Default()->NowMicros();
    std::cout << "Cplex solver time: " << TIME_US_TO_MS(end_time - start_time)
              << "\n";

    return inputSize;
  }

private:
  void init(std::vector<double> &modelAccuracy) {
    // create model
    model_m = IloModel(env_m);

    // create decision variables and value variables to store predicted
    // values
    constraints_m = IloConstraintArray(env_m);
    createVariables(modelAccuracy);

    // create constraints
    createConstraints();
    model_m.add(constraints_m);

    // create an objective function
    createObjective();
    model_m.add(objective_m);

    cplex_m = IloCP(model_m);

    // export model for debugging purpose
    cplex_m.exportModel("model.cpo");
  }

  void createVariables(std::vector<double> &modelAccuracy) {
    inputSize_m = IloIntVarArray(env_m, horizon_m);
    rttPredTime_m = IloArray<IloNumArray>(env_m, horizon_m);
    modelAccuracy_m = IloNumArray(env_m, totalModels_m);
    std::stringstream name;
    for (unsigned int horizon = 0; horizon < horizon_m; ++horizon) {
      name << "InputSizeHorizon_" << horizon;
      inputSize_m[horizon] =
          IloIntVar(env_m, 0, totalModels_m, name.str().c_str());
      rttPredTime_m[horizon] = IloNumArray(env_m, totalModels_m);
      name.str("");
    }

    for (unsigned int model = 0; model < totalModels_m; ++model) {
      modelAccuracy_m[model] = modelAccuracy[model];
    }
  }

  void createConstraints() {
    // Constraint 1
    // Input size should be within the expected values
    // This is already added in the variable declaration

    // Constraint 2
    // RTT should be below threshold. TODO: This is currently in the
    // constraint and can be part of objective function to minimize the
    // difference between RTT and set point. Type:1

    constraints_m.clear();
    for (unsigned int horizon = 0; horizon < horizon_m; ++horizon) {
      constraints_m.add(rttPredTime_m[horizon][inputSize_m[horizon]] <=
                        RTT_THRESHOLD);
    }
  }

  void createObjective() {
    IloExpr expr_m(env_m);
    // accuracy error cost
    for (unsigned int horizon = 0; horizon < horizon_m; ++horizon) {
      expr_m += weightForAccuracy_m *
                IloSquare(100.0 - modelAccuracy_m[inputSize_m[horizon]]);
    }

    // switching cost
    double weight_switching = 100; // get from global config object
    for (unsigned int horizon = 0; horizon < horizon_m; ++horizon) {
      if (horizon == 0) {
        expr_m += weightForSwitching_m *
                  IloSquare(prevInputSize - inputSize_m[horizon]);
      } else {
        expr_m += weight_switching *
                  IloSquare(inputSize_m[horizon - 1] - inputSize_m[horizon]);
      }
    }

    objective_m = IloObjective(env_m, expr_m, IloObjective::Minimize);
  }

  const int RTT_THRESHOLD =
      33; // TODO: We have to move this to a global config object.
  IloEnv env_m;
  IloModel model_m;
  IloConstraintArray constraints_m;
  IloObjective objective_m;
  IloCP cplex_m;
  IloArray<IloNumArray> rttPredTime_m;
  IloNumArray modelAccuracy_m;
  IloIntVarArray inputSize_m;
};

Solver *Solver::createInstance(unsigned int horizon, unsigned int totalModels,
                               std::vector<double> modelsAccuracy,
                               unsigned int defaultInputSize,
                               double weightForAccuracy,
                               double weightForSwitching) {
  Solver *_instance =
      new CplexSolver(horizon, totalModels, modelsAccuracy, defaultInputSize,
                      weightForAccuracy, weightForSwitching);

  return _instance;
}

#else

#include <glpk.h>
class GlpkSolver : public Solver {
public:
  explicit GlpkSolver(unsigned int horizon, unsigned int totalModels,
                      std::vector<double> modelAccuracy,
                      unsigned int defaultInputSize, double weightForAccuracy,
                      double weightForSwitching)
      : Solver(horizon, totalModels, modelAccuracy, defaultInputSize,
               weightForAccuracy, weightForSwitching) {
    init();
  }

  ~GlpkSolver() {
    glp_delete_prob(prob_m);
    deleteParam();
  }

  std::vector<unsigned int>
  solve(const std::vector<double> &throughputHorizon,
        const std::vector<std::vector<double>> &modelTimeHorizon,
        const std::vector<std::vector<size_t>> &netSizeHorizon,
        double lastModelAccuracy, double lastEndToEndTime,
        std::vector<double> &endToEndTime) {

    auto start_time = EnvTime::Default()->NowMicros();

    // lastEndToEndTime = 0.0;
    // compute RTT and Set model time and network time values
    for (unsigned int horizon = 0; horizon < horizon_m; ++horizon) {
      // std::cout << "\nHORIZON: " << horizon << std::endl;
      auto modelTime_len = modelTime_param_m[horizon][0][2];
      auto modelTime_row = modelTime_param_m[horizon][0][0];
      int modelTime_indexes[modelTime_len] = {0};
      double modelTime_values[modelTime_len] = {1};
      assert(modelTime_len == (totalModels_m + 1));

      auto networkTime_len = networkTime_param_m[horizon][0][2];
      auto networkTime_row = networkTime_param_m[horizon][0][0];
      int networkTime_indexes[networkTime_len] = {0};
      double networkTime_values[networkTime_len] = {0};
      assert(networkTime_len == (totalModels_m + 1));

      for (unsigned int model = 0; model <= totalModels_m; ++model) {
        auto rtt =
            getRTT(throughputHorizon[horizon], modelTimeHorizon[horizon][model],
                   netSizeHorizon[horizon][model]);
        if (rtt == std::numeric_limits<double>::infinity()) {
          rtt = 1000000;
        }
        modelTime_indexes[model + 1] = modelTime_param_m[horizon][model][1];
        networkTime_indexes[model + 1] = networkTime_param_m[horizon][model][1];

        if (model == totalModels_m) {
          modelTime_values[model + 1] = 1;
          networkTime_values[model + 1] = 1;
        } else {
          modelTime_values[model + 1] = -(modelTimeHorizon[horizon][model]);
          networkTime_values[model + 1] =
              -(rtt - modelTimeHorizon[horizon][model]);
        }
        // std::cout << "\t ModelPred Time: " << modelTime_indexes[model + 1]
        //           << " " << modelTime_values[model + 1] << std::endl;
        // std::cout << "\t Network Time: " << networkTime_indexes[model + 1]
        //           << " " << networkTime_values[model + 1] << std::endl;
      }

      glp_set_mat_row(prob_m, modelTime_row, modelTime_len, modelTime_indexes,
                      modelTime_values);
      glp_set_mat_row(prob_m, networkTime_row, networkTime_len,
                      networkTime_indexes, networkTime_values);
    }
    std::cout << "\n";

    // set last model accuracy
    glp_set_row_bnds(prob_m, last_modelAcc_idx_m, GLP_FX, lastModelAccuracy,
                     lastModelAccuracy);
    glp_set_row_bnds(prob_m, last_endToEnd_idx_m, GLP_FX, lastEndToEndTime,
                     lastEndToEndTime);

    bool solved = false;
    std::vector<unsigned int> inputSize;
    int horizon_idx = 0, models_idx = 0;
    int e2e_horizon_idx = 0;
    start_time = EnvTime::Default()->NowMicros();
    auto ret = glp_intopt(prob_m, &iocp_m);
    if (ret == 0 && glp_mip_status(prob_m) == GLP_OPT) {
      auto columns = glp_get_num_cols(prob_m);
      for (auto i = 1; i <= columns; ++i) {
        std::string colName(glp_get_col_name(prob_m, i));
        std::stringstream ss;
        ss << "control_model(" << (horizon_idx + 1) << "," << (models_idx + 1)
           << ")";
        std::stringstream endToEndSS;
        endToEndSS << "endToEnd_time(" << (e2e_horizon_idx + 1) << ")";
        std::string serverWaitSS = "server_queue_wait(1)";
        if (ss.str().compare(colName) == 0) {
          auto value = glp_mip_col_val(prob_m, i);
          if (value == 1) {
            inputSize.push_back(models_idx);
          }
          models_idx = (models_idx + 1) % totalModels_m;
          horizon_idx = models_idx ? horizon_idx : (horizon_idx + 1);
        } else if (endToEndSS.str().compare(colName) == 0) {
          auto value = glp_mip_col_val(prob_m, i);
          // std::cout << "End2Endtime: " << e2e_horizon_idx << " " << value << std::endl;
          endToEndTime.push_back(value);
          ++e2e_horizon_idx;
        } else if (serverWaitSS.compare(colName) == 0) {
          auto value = glp_mip_col_val(prob_m, i); 
          // std::cout << "ServerWaitQueue: " << 1 << " " << value << std::endl;
        }
      }
      assert(inputSize.size() == horizon_m);
      assert(endToEndTime.size() == horizon_m);
    } else {
      std::cout << "Infeasible GLPK Solution, Type: " << ret << std::endl;
    }
    auto end_time = EnvTime::Default()->NowMicros();
    std::cout << "GLPK solver time: " << TIME_US_TO_MS(end_time - start_time)
              << "\n";
    // glp_print_mip(prob_m, "temp.out");
    return inputSize;
  }

private:
  void init() {
    // create model
    prob_m = glp_create_prob();
    glp_init_iocp(&iocp_m);
    iocp_m.presolve = GLP_ON;
    iocp_m.msg_lev = GLP_MSG_OFF;

    // read model in LP format
    auto ret =
        glp_read_lp(prob_m, NULL, ConfigManager::Default()->glpkLpFile.c_str());

    if (ret != 0) {
      throw std::runtime_error("Cannot read LP file for GLPK solver ");
    }

    // record model time and network time param index of model so that we can
    // modify it later
    allocParam();

    getRowIndexes(modelTime_param_m, "def_selected_model_time");
    getRowIndexes(networkTime_param_m, "def_selected_network_time");
    getRowIndexes(&last_modelAcc_idx_m, "def_last_model_accuracy");
    getRowIndexes(&last_endToEnd_idx_m, "def_last_endToEnd_time");
  }

  void getRowIndexes(int ***row_idx, std::string name) {
    auto columns = glp_get_num_cols(prob_m);
    auto rows = glp_get_num_rows(prob_m);
    std::vector<int> ind;
    std::vector<double> val;
    ind.resize(columns + 1);
    val.resize(columns + 1);

    int horizon_idx = 0, models_idx = 0;
    for (auto row = 1; row <= rows; ++row) {
      auto rowName = glp_get_row_name(prob_m, row);
      auto type = glp_get_row_type(prob_m, row);
      std::stringstream ss_rtt;
      ss_rtt << name << "(" << (horizon_idx + 1) << ")";

      if (type == GLP_FX && ss_rtt.str().compare(rowName) == 0) {

        std::cout << "RowName: " << rowName << " Kind: " << type << std::endl;

        auto len = glp_get_mat_row(prob_m, row, ind.data(), val.data());
        assert(len == (totalModels_m + 1));

        for (auto model = 1; model < (len + 1); ++model) {
          std::cout << "\tIndex: " << ind[model] << " Value: " << val[model]
                    << std::endl;
          row_idx[horizon_idx][model - 1][0] = row;
          row_idx[horizon_idx][model - 1][1] = ind[model];
          row_idx[horizon_idx][model - 1][2] = len;
        }
        ++horizon_idx;
      }
    }
  }

  void getRowIndexes(int *row_idx, std::string name) {
    auto columns = glp_get_num_cols(prob_m);
    auto rows = glp_get_num_rows(prob_m);
    std::vector<int> ind;
    std::vector<double> val;
    ind.resize(columns + 1);
    val.resize(columns + 1);

    int horizon_idx = 0, models_idx = 0;
    for (auto row = 1; row <= rows; ++row) {
      auto rowName = glp_get_row_name(prob_m, row);
      auto type = glp_get_row_type(prob_m, row);

      if (type == GLP_FX && name.compare(rowName) == 0) {
        std::cout << "RowName: " << rowName << " Kind: " << type << std::endl;
        auto ub = glp_get_row_ub(prob_m, row);
        auto lb = glp_get_row_lb(prob_m, row);
        std::cout << "\tIndex: "
                  << " Ub: " << ub << " Lb: " << lb << std::endl;
        *row_idx = row;
      }
    }
  }

  void allocParam() {
    modelTime_param_m = new int **[horizon_m];
    networkTime_param_m = new int **[horizon_m];
    for (int i = 0; i < horizon_m; i++) {
      modelTime_param_m[i] = new int *[totalModels_m + 1];
      networkTime_param_m[i] = new int *[totalModels_m + 1];
      for (int j = 0; j <= totalModels_m; j++) {
        modelTime_param_m[i][j] = new int[3];
        networkTime_param_m[i][j] = new int[3];
      }
    }
  }

  void deleteParam() {
    // deallocate memory
    for (int i = 0; i < horizon_m; i++) {
      for (int j = 0; j <= totalModels_m; j++) {
        delete[] modelTime_param_m[i][j];
        delete[] networkTime_param_m[i][j];
      }
      delete[] modelTime_param_m[i];
      delete[] networkTime_param_m[i];
    }

    delete[] modelTime_param_m;
    delete[] networkTime_param_m;
  }

  glp_prob *prob_m;
  glp_iocp iocp_m;
  int ***modelTime_param_m;   // [row, index, len]
  int ***networkTime_param_m; // [row, index, len]
  int last_modelAcc_idx_m;
  int last_endToEnd_idx_m;
};

Solver *Solver::createInstance(unsigned int horizon, unsigned int totalModels,
                               std::vector<double> modelsAccuracy,
                               unsigned int defaultInputSize,
                               double weightForAccuracy,
                               double weightForSwitching) {
  Solver *_instance =
      new GlpkSolver(horizon, totalModels, modelsAccuracy, defaultInputSize,
                     weightForAccuracy, weightForSwitching);

  return _instance;
}

#endif /* solver type */

#endif /*  __MPC_SOLVER_H__ */
