#include <assert.h>
#include <cstring>
#include <glpk.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

typedef unsigned long long uint64;
#include "env_time.h"

#define TOTAL_MODELS 5
#define RTT_THRESHOLD 30
#define MPC_HORIZON 5

double modelAccuracy[TOTAL_MODELS] = {20.0, 35.0, 45.0, 50.0, 51.0};
double inputSize[TOTAL_MODELS] = {16, 32, 128, 256, 1024};
double modelPredTime[TOTAL_MODELS] = {2, 4, 8, 10, 12};
double predThroughput[MPC_HORIZON] = {100.0, 50.0, 30.0, 80.0, 100.0};
double predThroughput1[MPC_HORIZON] = {0.1, 1.0, 1.0, 1.0, 0.0};

double getPredictedRTT(const unsigned int model,
                       const unsigned int throughput) {
  auto _inputSize =
      inputSize[model]; // This has to be encoded input size. TODO:
                        // Must include object detection size
  double rtt = (_inputSize / throughput) + modelPredTime[model];

  return rtt;
}

void glpk_solve(glp_prob *lp, glp_iocp *iocp,
                int modelTime_row_idx[MPC_HORIZON][TOTAL_MODELS + 1][3],
                int networkTime_row_idx[MPC_HORIZON][TOTAL_MODELS + 1][3],
                int last_model_accuracy_idx, int last_endToend_idx,
                double last_model_accuracy, double last_endToend_time,
                double throughput[MPC_HORIZON]) {
  auto start_time = EnvTime::Default()->NowMicros();

  // Set model time and network time values
  for (unsigned int horizon = 0; horizon < MPC_HORIZON; ++horizon) {
    std::cout << "\nHORIZON: " << horizon << std::endl;
    auto modelTime_len = modelTime_row_idx[horizon][0][2];
    auto modelTime_row = modelTime_row_idx[horizon][0][0];
    int modelTime_indexes[modelTime_len] = {0};
    double modelTime_values[modelTime_len] = {1};
    assert(modelTime_len == (TOTAL_MODELS + 1));

    auto networkTime_len = networkTime_row_idx[horizon][0][2];
    auto networkTime_row = networkTime_row_idx[horizon][0][0];
    int networkTime_indexes[networkTime_len] = {0};
    double networkTime_values[networkTime_len] = {0};
    assert(networkTime_len == (TOTAL_MODELS + 1));

    for (unsigned int model = 0; model <= TOTAL_MODELS; ++model) {
      auto rtt = getPredictedRTT(model, throughput[horizon]);
      if (rtt == std::numeric_limits<double>::infinity()) {
        rtt = 1000000;
      }
      modelTime_indexes[model + 1] = modelTime_row_idx[horizon][model][1];
      networkTime_indexes[model + 1] = networkTime_row_idx[horizon][model][1];

      if (model == TOTAL_MODELS) {
        modelTime_values[model + 1] = 1;
        networkTime_values[model + 1] = 1;
      } else {
        modelTime_values[model + 1] = -(modelPredTime[model]);
        networkTime_values[model + 1] = -(rtt - modelPredTime[model]);
      }
      std::cout << "\t ModelPred Time: " << modelTime_indexes[model + 1] << " "
                << modelTime_values[model + 1] << std::endl;
      std::cout << "\t Network Time: " << networkTime_indexes[model + 1] << " "
                << networkTime_values[model + 1] << std::endl;
    }

    glp_set_mat_row(lp, modelTime_row, modelTime_len, modelTime_indexes,
                    modelTime_values);
    glp_set_mat_row(lp, networkTime_row, networkTime_len, networkTime_indexes,
                    networkTime_values);
  }

  // Set last model accuracy
  glp_set_row_bnds(lp, last_model_accuracy_idx, GLP_FX, last_model_accuracy,
                   last_model_accuracy);

  // set last model pred time
  glp_set_row_bnds(lp, last_endToend_idx, GLP_FX, last_endToend_time,
                   last_endToend_time);

  glp_write_lp(lp, NULL, "temp.lp");
  auto ret = glp_intopt(lp, iocp);
  if (ret == 0 && glp_mip_status(lp) == GLP_OPT) {
    auto columns = glp_get_num_cols(lp);
    int horizon_idx = 0, models_idx = 0;
    int e2e_horizon_idx = 0;
    for (auto i = 1; i <= columns; ++i) {
      auto type = glp_get_col_type(lp, i);
      std::string colName(glp_get_col_name(lp, i));
      std::stringstream ss;
      ss << "control_model(" << (horizon_idx + 1) << "," << (models_idx + 1)
         << ")";
      std::stringstream endToEndSS;
      endToEndSS << "endToEnd_time(" << (e2e_horizon_idx + 1) << ")";
      if (ss.str().compare(colName) == 0) {
        auto value = glp_mip_col_val(lp, i);
        if (value == 1) {
          std::cout << "Found Model: " << models_idx << std::endl;
        }
        std::cout << colName << "\tValue: " << value << "\ttype: " << type
                  << std::endl;
        models_idx = (models_idx + 1) % TOTAL_MODELS;
        horizon_idx = models_idx ? horizon_idx : (horizon_idx + 1);
      } else if (endToEndSS.str().compare(colName) == 0) {
        auto value = glp_mip_col_val(lp, i);
        std::cout << colName << "\tValue: " << value << "\ttype: " << type
                  << std::endl;
        ++e2e_horizon_idx;
      }
    }
  } else {
    std::cout << "Infeasible Solution Type: " << ret << std::endl;
  }

  auto end_time = EnvTime::Default()->NowMicros();
  std::cout << "Time Taken: " << TIME_US_TO_MS(end_time - start_time)
            << std::endl;
}

void getRowIndexes(glp_prob *lp, int row_idx[MPC_HORIZON][TOTAL_MODELS + 1][3],
                   std::string name) {
  auto columns = glp_get_num_cols(lp);
  auto rows = glp_get_num_rows(lp);
  std::vector<int> ind;
  std::vector<double> val;
  ind.resize(columns + 1);
  val.resize(columns + 1);

  int horizon_idx = 0, models_idx = 0;
  for (auto row = 1; row <= rows; ++row) {
    auto rowName = glp_get_row_name(lp, row);
    auto type = glp_get_row_type(lp, row);
    std::stringstream ss_rtt;
    ss_rtt << name << "(" << (horizon_idx + 1) << ")";

    if (type == GLP_FX && ss_rtt.str().compare(rowName) == 0) {

      std::cout << "RowName: " << rowName << " Kind: " << type << std::endl;

      auto len = glp_get_mat_row(lp, row, ind.data(), val.data());
      assert(len == (TOTAL_MODELS + 1));

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

void getRowIndexes(glp_prob *lp, int *row_idx, std::string name) {
  auto columns = glp_get_num_cols(lp);
  auto rows = glp_get_num_rows(lp);
  std::vector<int> ind;
  std::vector<double> val;
  ind.resize(columns + 1);
  val.resize(columns + 1);

  int horizon_idx = 0, models_idx = 0;
  for (auto row = 1; row <= rows; ++row) {
    auto rowName = glp_get_row_name(lp, row);
    auto type = glp_get_row_type(lp, row);

    if (type == GLP_FX && name.compare(rowName) == 0) {
      std::cout << "RowName: " << rowName << " Kind: " << type << std::endl;
      auto ub = glp_get_row_ub(lp, row);
      auto lb = glp_get_row_lb(lp, row);
      std::cout << "\tIndex: "
                << " Ub: " << ub << " Lb: " << lb << std::endl;
      *row_idx = row;
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr << "Usage: ./test_glpk <lp file>" << std::endl;
    exit(EXIT_FAILURE);
  }

  glp_prob *lp;

  // read cplex LP file
  lp = glp_create_prob();
  glp_iocp iocp;
  glp_init_iocp(&iocp);
  iocp.presolve = GLP_ON;
  iocp.msg_lev = GLP_MSG_OFF;

  glp_set_prob_name(lp, "mpc_controller");
  auto ret = glp_read_lp(lp, NULL, argv[1]);

  if (ret != 0) {
    std::cerr << "Cannot read LP file " << std::endl;
    glp_delete_prob(lp);
    exit(EXIT_FAILURE);
  }

  // ret = glp_intopt(lp, &iocp);
  // glp_print_mip(lp, "temp.out");

  int modelTime_row_indexes[MPC_HORIZON][TOTAL_MODELS + 1][3] = {-1};
  int networkTime_row_indexes[MPC_HORIZON][TOTAL_MODELS + 1][3] = {-1};
  /* TODO: Combine multiple function calls into one if it get's too expensive */
  getRowIndexes(lp, modelTime_row_indexes, "def_selected_model_time");
  getRowIndexes(lp, networkTime_row_indexes, "def_selected_network_time");

  int last_model_acc_idx = 1;
  int last_endToend_idx = 1;
  getRowIndexes(lp, &last_model_acc_idx, "def_last_model_accuracy");
  getRowIndexes(lp, &last_endToend_idx, "def_last_endToEnd_time");

  double last_endToend_time = 0;
  // solve
  glpk_solve(lp, &iocp, modelTime_row_indexes, networkTime_row_indexes,
             last_model_acc_idx, last_endToend_idx, modelAccuracy[0],
             last_endToend_time, predThroughput);
  glpk_solve(lp, &iocp, modelTime_row_indexes, networkTime_row_indexes,
             last_model_acc_idx, last_endToend_idx, modelAccuracy[1],
             last_endToend_time, predThroughput1);
  last_endToend_time = 60; // 70 ms does not have a solution
  glpk_solve(lp, &iocp, modelTime_row_indexes, networkTime_row_indexes,
             last_model_acc_idx, last_endToend_idx, modelAccuracy[2],
             last_endToend_time, predThroughput);
  // glp_print_mip(lp, "temp.out");
  // glp_write_lp(lp, NULL, "temp.lp");
  glp_delete_prob(lp);
}
