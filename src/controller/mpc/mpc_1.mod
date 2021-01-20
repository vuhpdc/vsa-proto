/* MPC controller to predict optimized value of next H (horizon) frame input size */
using cp;
/* Define constant params */
param HORIZON, integer, >= 1;
param MODELS, integer, >= 1;
param ACC_WEIGHT, default 1.0;
param SWITCHING_WEIGHT, default 0.0;
param RTT_THRESHOLD, integer, default 33;

/* Define indexes */
set horizon := 1..HORIZON;
set models := 1..MODELS;


/* Define data params */
param rtt{horizon, models}; /* Predicted values of RTT in ms for next h horizon values */
param model_pred_time{horizon, models}; /* Predicted values of m models prediction time on server for next h horizon values */
param model_error{models}; /* Validation accuracy of m models */

/* Objective function to optimize quality of analytics */
/* TODO: We cannot use decision variable as a index in param, unlike IBM CP */ 
var input_size{horizon}, integer, >= 0, <= MODELS;
# minimize QoA:	sum{h in horizon} (ACC_WEIGHT * model_error[input_size[h]])
#  + sum{h in 1..(HORIZON-1)} (SWITCHING_WEIGHT * (input_size[h+1] - input_size[h]));

/* Constraints */
s.t. rtt_constraint{h in horizon}: rtt[input_size[h]] <= RTT_THRESHOLD;
