/* MPC controller to predict optimized value of next H (horizon) frame input size */

/* Define constant params */
param HORIZON, integer, >= 1, default 5;
param MODELS, integer, >= 1, default 5;
param ACC_WEIGHT, default 1.0;
param SWITCHING_WEIGHT, default 1.0;
param RTT_THRESHOLD, integer, default 33;
param FRAME_INTER_ARRIVAL, default 33;

/* Define indexes */
set horizon := 1..HORIZON;
set models := 1..MODELS;

/* Define data params */
param model_accuracy{models}, default 50.0; /* Validation accuracy of m models */
param last_model, integer, default 1; /* Model used in the last horizon */
param model_prof_time{models}, default 1.0; /* Profiled values of Model detection time on the server */
param slowdown_factor_param, default 1.0; /* Slow down factor to capture server load */
param last_endToEnd_time_param, default 1.0; /* EndToEnd time of the last frame */


/* Define decision variable */
var control_model{horizon, models}, binary; /* control_model[h, m] = 1, when model m is used for horizon h */
s.t. one{h in horizon}: sum{m in models} control_model[h, m] = 1;
/* Only one model is used per horizon */


/* Get the control model accuracy */
var control_model_accuracy{h in horizon} >= 0.0;
s.t. def_control_model_accuracy{h in horizon}:
    control_model_accuracy[h] = sum{m in models}model_accuracy[m]*control_model[h,m];


/* Absolute smoothness constraint in a linear format */
var abs_smoothness{horizon};
var last_model_accuracy >= 0.0;

s.t. def_last_model_accuracy:
    last_model_accuracy = model_accuracy[last_model];

s.t. last_abs_smoothness_1: (last_model_accuracy - control_model_accuracy[1]) <= abs_smoothness[1];
s.t. last_abs_smoothness_2: -(last_model_accuracy - control_model_accuracy[1]) <= abs_smoothness[1];
s.t. remain_abs_smoothness_1{h in 2..HORIZON}:
      (control_model_accuracy[h-1] - control_model_accuracy[h]) <= abs_smoothness[h];
s.t. remain_abs_smoothness_2{h in 2..HORIZON}:
      -(control_model_accuracy[h-1] - control_model_accuracy[h]) <= abs_smoothness[h];
/* END: Absolute smoothness constaint */

   
/* model prediction time with slowdown factor */
# var slowdown_factor, >= 0.0;
# s.t. def_slowdown_factor:
#    slowdown_factor = slowdown_factor_param;
# var model_time{models}, >= 0.0;
# s.t. def_model_time{m in models}:
#    model_time[m] = slowdown_factor * model_prof_time[m];
param model_time{models}, default 1.5;
var selected_model_time{horizon}, >= 0.0;
s.t. def_selected_model_time{h in horizon}:
    selected_model_time[h] = sum{m in models} model_time[m] * control_model[h, m];


/* Network size and time */
# var network_throughput{horizon}, >= 1.0;
# s.t. def_network_throughput{h in horizon}:
#    network_throughput[h] = network_throughput_param[h];
# var network_size{horizon, models};
# s.t. def_network_size{h in horizon, m in models}:
#     network_size[h, m] = network_size_param[h,m];
# var network_time{horizon, models};
# s.t. def_network_time{h in horizon, m in models}:
#    network_time[h, m] = (network_size[h, m] / network_throughput[h]) * 8; /* in ms */
param network_time{horizon, models}, default 1.5;
var selected_network_time{horizon}, >= 0.0;
s.t. def_selected_network_time{h in horizon}:
     selected_network_time[h] = sum{m in models} network_time[h, m] * control_model[h, m];


/* EndToEnd time */
var endToEnd_time{horizon}, >= 0.0, <= RTT_THRESHOLD;


/* START: Server queue waiting time */
var last_endToEnd_time, >= 0.0;
s.t. def_last_endToEnd_time:
    last_endToEnd_time = last_endToEnd_time_param;
var server_queue_wait{horizon}, >=0;
var s_binary{horizon}, binary; /* 0 for positive, 1 for negative */
param s_UP, default 150.0;
param s_LOW, default -150.0;

s.t. first_server_queue_wait_1:
    s_LOW * s_binary[1] <= (last_endToEnd_time - FRAME_INTER_ARRIVAL - selected_network_time[1]);
s.t. first_server_queue_wait_2:
    (last_endToEnd_time - FRAME_INTER_ARRIVAL - selected_network_time[1]) <= s_UP * (1 - s_binary[1]);
s.t. first_server_queue_wait_3:
    server_queue_wait[1] <= (last_endToEnd_time - FRAME_INTER_ARRIVAL - selected_network_time[1]) - (s_LOW * s_binary[1]);
s.t. first_server_queue_wait_4:
    server_queue_wait[1] <= s_UP * (1 - s_binary[1]);
s.t. first_server_queue_wait:
    server_queue_wait[1] >= last_endToEnd_time - FRAME_INTER_ARRIVAL - selected_network_time[1];

s.t. def_server_queue_wait_1{h in 2..HORIZON}:
    s_LOW * s_binary[h] <= (endToEnd_time[h-1] - FRAME_INTER_ARRIVAL - selected_network_time[h]);
s.t. def_server_queue_wait_2{h in 2..HORIZON}:
    (endToEnd_time[h-1] - FRAME_INTER_ARRIVAL - selected_network_time[h]) <= s_UP * (1 - s_binary[h]);
s.t. def_server_queue_wait_3{h in 2..HORIZON}:
    server_queue_wait[h] <= (endToEnd_time[h-1] - FRAME_INTER_ARRIVAL - selected_network_time[h]) - (s_LOW * s_binary[h]);
s.t. def_server_queue_wait_4{h in 2..HORIZON}:
    server_queue_wait[h] <= s_UP * (1 - s_binary[h]);
s.t. def_server_queue_wait{h in 2..HORIZON}:
    server_queue_wait[h] >= endToEnd_time[h-1] - FRAME_INTER_ARRIVAL - selected_network_time[h];

/* END: Waiting time linearization. max(waiting_time,0). Thanks to this blog: https://orinanobworld.blogspot.com/2010/12/lps-and-positive-part.html */

/* EndToEnd time */
s.t. def_endToEnd_time{h in horizon}:
    endToEnd_time[h] = selected_network_time[h] + selected_model_time[h] + server_queue_wait[h];


/* Define objectives */
var obj_accuracy >= 0.0;
s.t. def_obj_accuracy:
    obj_accuracy = sum{h in horizon} control_model_accuracy[h];

/* Overall objective function to optimize quality of analytics */
maximize QoA:
    ACC_WEIGHT * obj_accuracy - SWITCHING_WEIGHT * sum{h in horizon} abs_smoothness[h];

end;
