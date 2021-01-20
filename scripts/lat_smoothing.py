import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_opts():
    parser = argparse.ArgumentParser(
        description="Read raw latencies and estimate smooth value")

    parser.add_argument("--raw_txt_file", type=str,
                        help='File containing raw measurements')

    args = parser.parse_args()
    args_dict = args.__dict__
    print("------------------------------------")
    print("Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")

    return args


def read_latancies(csv_file):
    # read csv file with the following format
    # ''' Simulated Bandwidth (Mbit), Frame_id, Model_id_Frame_size,
    # Model_id_Used_detection_model, End-to-end_latency, Client_preprocess_latency,
    # Server_preprocess_latency, Server_detection_latency '''
    data = pd.read_csv(csv_file, delimiter=',', header=None)
    end2end_latencies = {}
    model_pred_latencies = {}
    for i in range(data.shape[0]):
        row = data.loc[i, :]
        try:
            model = int(row[3])
            end2end_latency = float(row[4]) - float(row[5])
            model_pred_latency = float(row[6]) + float(row[7])
        except:
            continue
        if model not in end2end_latencies:
            end2end_latencies[model] = []
            model_pred_latencies[model] = []

        end2end_latencies[model].append(end2end_latency)
        model_pred_latencies[model].append(model_pred_latency)

    # print mean and variance for each model
    for model in end2end_latencies.keys():
        lat_vec = np.asarray(end2end_latencies[model])
        model_pred_lat = np.asarray(model_pred_latencies[model])
        print("Model: ", model,
              "Latency(mean/var)={}/{}".format(np.mean(lat_vec),
                                               np.var(lat_vec)),
              "ModelPred(mean/var)={}/{}".format(np.mean(model_pred_lat), np.var(model_pred_lat)), len(lat_vec))
    return end2end_latencies, model_pred_latencies


def emwa(latencies, alpha=0.5):
    for i in range(1, len(latencies)):
        latencies[i] = (1 - alpha) * latencies[i] + alpha * latencies[i-1]

    return latencies


def stability_filter(latencies):
    latencies = np.asarray(latencies)
    instability_vec = np.zeros(len(latencies))

    # compute instability
    instability_vec[1] = abs(latencies[1] - latencies[0])
    beta = 0.6
    for i in range(2, len(latencies)):
        instability = abs(latencies[i]-latencies[i-1])
        instability_vec[i] = beta * \
            instability_vec[i-1] + (1 - beta) * instability

    # Compute smooth estimates
    for i in range(1, len(latencies)):
        if (i - 10) < 0:
            instability_max = np.max(instability_vec[0:i+1])
        else:
            instability_max = np.max(instability_vec[i-9:i+1])
        alpha = instability_vec[i] / instability_max
        latencies[i] = (1 - alpha) * latencies[i] + alpha * latencies[i-1]

    return latencies


def error_filter(latencies):
    latencies = np.asarray(latencies)
    error_vec = np.zeros(len(latencies))

    # Compute error
    error_vec[1] = abs(latencies[1] - latencies[0])
    gamma = 0.6
    for i in range(1, len(latencies)):
        if (i - 10) < 0:
            error_max = np.max(error_vec[0:i+1])
        else:
            error_max = np.max(error_vec[i-9:i+1])
        alpha = error_vec[i]/error_max
        latencies[i] = (1 - alpha) * latencies[i] + alpha * latencies[i-1]

        if i < (len(latencies)-1):
            error_vec[i+1] = gamma * error_vec[i] + \
                (1-gamma) * abs(latencies[i] - latencies[i+1])

    return latencies


def smooth_latencies(end2end_latency):
    _METHOD = "ERROR"
    if _METHOD == 'EMWA':
        end2end_latency = emwa(end2end_latency, 0.5)
    elif _METHOD == "STABILITY":
        end2end_latency = stability_filter(end2end_latency)
    elif _METHOD == "ERROR":
        end2end_latency = error_filter(end2end_latency)
    return end2end_latency


def plot_latencies(end2end_latencies, model_latencies):
    network_latency = [
        x1 - x2 for (x1, x2) in zip(end2end_latencies[9], model_latencies[9])]

    network_latency_smooth = network_latency[:]
    network_latency_smooth = smooth_latencies(network_latency_smooth)

    network_latency = np.asarray(network_latency)
    network_latency_smooth = np.asarray(network_latency_smooth)
    print("Mean/Var", "Network ({}/{})".format(np.mean(network_latency), np.var(network_latency)),
          "Network smooth ({}/{})".format(np.mean(network_latency_smooth), np.var(network_latency_smooth)))

    plt.plot(network_latency)
    plt.plot(network_latency_smooth)
    plt.legend(['Raw Network', 'Smooth Network'])
    plt.ylabel('Latency (ms)')
    # plt.ylim([6, 10])
    # plt.xlim([0, 200])
    plt.show()


def main():
    opts = parse_opts()

    # read latency measurements
    end2end_latencies, model_latencies = read_latancies(opts.raw_txt_file)

    # plot latencies
    plot_latencies(end2end_latencies, model_latencies)


if __name__ == '__main__':
    main()
