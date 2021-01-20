

// Magic tricks to have CPLEX behave well:
#ifndef IL_STD
#define IL_STD
#endif
#include <cstring>
// #include <ilcplex/ilocplex.h>
// ILOSTLBEGIN
// End magic tricks
typedef unsigned long long uint64;

#include "env_time.h"
#include <ilcp/cp.h>

#define TOTAL_MODELS 5
#define RTT_THRESHOLD 33
#define MPC_HORIZON 5

double modelAccuracy[TOTAL_MODELS] = {20.1, 35.1, 45.1, 50.1, 51.1};
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

void solve(const IloCP &cplex, const IloIntVarArray &input_size) {
  try {
    std::cout << "\n Calling solver.." << std::endl;

    // Try to solve with CPLEX (and hope it does not raise an exception!)
    auto start_time = EnvTime::Default()->NowMicros();
    bool solved = cplex.solve();
    auto end_time = EnvTime::Default()->NowMicros();
    if (solved) {
      // If CPLEX successfully solved the model, print the results
      std::cout << "\n\nCplex success!\n";
      std::cout << "\tStatus: " << cplex.getStatus() << "\n";
      std::cout << "\tObjective value: " << cplex.getObjValue() << "\n";
      std::cout << "\tTimeTaken: " << TIME_US_TO_MS(end_time - start_time)
                << "\n";
      for (unsigned int horizon = 0; horizon < MPC_HORIZON; ++horizon) {
        std::cout << "\tHorizon:" << horizon
                  << "\tInput value: " << cplex.getValue(input_size[horizon])
                  << "\n";
      }
    } else {
      std::cerr << "\n\nCplex error!\n";
      std::cerr << "\tStatus: " << cplex.getStatus() << "\n";
    }
  } catch (const IloException &e) {
    std::cerr << "\n\nCPLEX Raised an exception:\n";
    std::cerr << e << "\n";
  } catch (...) {
    std::cerr << "The following unknown exception was found: " << std::endl;
  }
}
int main() {

  IloEnv env;

  auto start_time = EnvTime::Default()->NowMicros();
  // define model
  IloModel model(env);

  // Get the measured values, either fixed or variable
  std::cout << "\n Creating array of measured values..";
  IloArray<IloNumArray> rttPredTime(env, MPC_HORIZON);
  IloNumArray _modelAccuracy(env, TOTAL_MODELS);
  std::stringstream name;
  for (unsigned int horizon = 0; horizon < MPC_HORIZON; ++horizon) {
    std::cout << "\nHORIZON: " << horizon << std::endl;
    rttPredTime[horizon] = IloNumArray(env, TOTAL_MODELS);
    for (unsigned int model = 0; model < TOTAL_MODELS; ++model) {
      rttPredTime[horizon][model] =
          getPredictedRTT(model, predThroughput[horizon]);
      _modelAccuracy[model] = modelAccuracy[model];

      std::cout << "\n\tModel: " << model
                << "\tRTT: " << rttPredTime[horizon][model]
                << "\tModelAccuracy: " << _modelAccuracy[model] << std::endl;
    }
  }

  // define variables. This includes control decision variable
  std::cout << "\n Creating decision variables..";
  IloIntVarArray input_size(env, MPC_HORIZON);
  for (unsigned int i = 0; i < MPC_HORIZON; ++i) {
    name << "InputSizeHorizon_" << i;
    input_size[i] = IloIntVar(env, 0, TOTAL_MODELS, name.str().c_str());
    name.str("");
  }

  // define constraints
  std::cout << "\n Creating constraints..";
  IloExpr expr(env);
  IloConstraintArray constraint(env);

  // Constraint 1
  // Input size should be within the expected values
  // This is already added in the variable declaration

  // Constraint 2
  // RTT should be below threshold. TODO: This is currently in the
  // constraint and can be part of objective function to minimize the difference
  // between RTT and set point.
  // Type:1
  for (unsigned int horizon = 0; horizon < MPC_HORIZON; ++horizon) {
    // expr = rttPredTime[horizon][input_size[horizon]];
    constraint.add(rttPredTime[horizon][input_size[horizon]] <= RTT_THRESHOLD);
    // model.add(expr <= RTT_THRESHOLD);
    // model.add(rttPredTime[horizon][input_size[horizon]] <= RTT_THRESHOLD);
    // expr.clear();
  }
  model.add(constraint);

  // define objective function
  std::cout << "\n Creating objectives..";
  expr.clear();
  // accuracy error cost
  double weight_accuracy = 1.0;
  for (unsigned int horizon = 0; horizon < MPC_HORIZON; ++horizon) {
    // expr += weight_accuracy *
    //         IloSquare(100.0 - _modelAccuracy[input_size[horizon]]);
    expr += weight_accuracy * _modelAccuracy[input_size[horizon]];
  }

  // switching cost
  double weight_switching = 0;
  double prev_input_size = 0;
  for (unsigned int horizon = 0; horizon < MPC_HORIZON; ++horizon) {
    if (horizon == 0) {
      expr +=
          weight_switching * IloSquare(prev_input_size - input_size[horizon]);
    } else {
      expr += weight_switching *
              IloSquare(input_size[horizon - 1] - input_size[horizon]);
    }
  }

  // NOTE: Add below term to the objective function
  //   double weight_rtt = 1.0;
  //   expr += weight_rtt * IloSquare(rttPredTime[input_size] -
  //   RTT_THRESHOLD);
  IloObjective obj(env, expr, IloObjective::Maximize);

  // Add the objective function to the model
  model.add(obj);

  // Free the memory used by expr
  expr.end();

  // Create the solver object
  // IloCplex cplex(model);
  IloCP cplex(model);

  auto end_time = EnvTime::Default()->NowMicros();
  std::cout << "\n Created cplex object.. "
            << TIME_US_TO_MS(end_time - start_time) << std::endl;

  // Export model to file (useful for debugging!)
  // cplex.exportModel("model.lp");
  cplex.exportModel("model.cpo");
  solve(cplex, input_size);

  // next iteration
  start_time = EnvTime::Default()->NowMicros();
  model.remove(constraint);
  constraint.clear();

  // test if we can use same model and constraints
  for (unsigned int horizon = 0; horizon < MPC_HORIZON; ++horizon) {
    std::cout << "\nHORIZON: " << horizon << std::endl;
    // rttPredTime[horizon] = IloNumArray(env, TOTAL_MODELS);
    for (unsigned int model = 0; model < TOTAL_MODELS; ++model) {
      rttPredTime[horizon][model] =
          getPredictedRTT(model, predThroughput1[horizon]);

      std::cout << " " << rttPredTime[horizon][model];
    }
    std::cout << "\n";
  }

  for (unsigned int horizon = 0; horizon < MPC_HORIZON; ++horizon) {
    constraint.add(rttPredTime[horizon][input_size[horizon]] <= RTT_THRESHOLD);
  }
  model.add(constraint);
  // cplex.extract(model); // expensive operation and not needed.
  end_time = EnvTime::Default()->NowMicros();
  std::cout << "\n ReCreated cplex object.. "
            << TIME_US_TO_MS(end_time - start_time) << std::endl;
  solve(cplex, input_size);

  env.end();
}