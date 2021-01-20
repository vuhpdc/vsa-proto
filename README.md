# Video Stream Analytics
A bare mimimum framework to carry out video analytics on a remote server by streaming video frames. It has been carved out from [video_lcp](https://github.com/vuhpdc/video_lcp.git). So, if something is missing in this repo then please refer back to `video_lcp` (and branch `vinod_develop`).

# Requirements
1. cmake >= 3.12
2. opencv == 4.4.0

# Getting started

* Checkout submodules, `darknet` and `object_detection_metrics`.
```bash
$ git submodule update --init --recursive
```

* Server module uses `darknet` as a DNN inference framework.
Note that darknet needs cmake version >= 3.12.
```bash
$ ./scripts/install_darknet.sh
```

* Download MS COCO image dataset and a video from PKUMMD video dataset.
```bash
$ ./scripts/datasets.sh
```

## Build
Build client and server module with the following commands.

```bash
$ mkdir build

$ cd build
```

1. Client
```bash
$ cmake ../ -DCLIENT=1
```
By default, the client module employs the fixed controller that streams video frames with fixed size resolution. To build the client with a different type of controller for frame size adaptation, add `-DCONTROLLER_TYPE="basic"` to the cmake command.

2. Server
```bash
$ cmake ../ -DSERVER=1
```

To build the test module, add `-DTEST=1` to the cmake command.

## Run
Run client and server with the following commands.
1. Client
```bash
$ ./client ../client_config.json
```
Modify [client_config.json](./client_config.json) file to change the configurations such as server hostname (IP address), port number, video path, etc. Check [src/common.h](./src/common.h) for configuration variables.

2. Server
```bash
$ ./server ../server_config.json 
```
We mostly run the server module on DAS-5. To compile the server on DAS-5, set up environment variables using the following command.
```bash
$ source ./scripts/env_das5.sh
```

We cannot directly connect to DAS-5 compute node, and we, therefore, have to relay traffic from the DAS-5 head node to a specific compute node. Use the following example command on the headnode,
```bash
$ socat TCP-LISTEN:10001,fork,reuseaddr,nodelay TCP:node029:10001,nodelay
```

## Measure accuracy
1. On COCO dataset

The inference results are saved in `output.json` file. Let's assume that we run the client module with COCO images by setting `image_list` in `config.json` file to `../datasets/coco/val2017/image_list.txt`

The following command calculates the `mean average precision (mAP)` for object detection task.
```bash
$ python3 ./metrics/CocoMap/Coco_mAP.py --annotation_file ./datasets/coco/val2017/annotations/instances_val2017.json --result_file ./build/output.json
```

For this, we need `pycocotools` which can be installed from [here](https://github.com/philferriere/cocoapi.git).

2. On PKUMMD video

Let's assume that we run the client module with PKUMMD video  by setting `video_path` in `config.json` file to `../datasets/pkummd/0200-M.avi`.

We have to first convert the inference results for the video saved in the `output.json` to the per-frame results. 
```bash
$ python3 ./metrics/CocoMap/output_json2txt.py --input_file ./build/output.json --output_dir results
```

The following command calculates the `mAP`,
```bash
$ python3 ./metrics/object_detection_metrics/pascalvoc.py -gt ./results_gt -det ./results -np
```
Note that `results_gt` has to be generated similar to the above `results` but by using highly accurate model for the ground truth.


## Network simulation
We use the `tc` utility to emulate different network scenarios. To enable traffic shaping, set `network_shaping` to 1 and `shaping_file` to the text file that contains network rate limits. For details, check out the [shape.sh](./simulation/shape.sh) file.

## Visualize
Currently, `client` module does not display video on the console. However, it contains untested code. 

On the other hand, we can visualize a video along with the inference results on the console. Use the following command, 
```bash
$ python3 scripts/visualize.py --video_file datasets/pkummd/0200-M.avi --annotation_gt_file ./build/output.json --annotation_det_file ./build/output.json --label_file ./datasets/coco/val2017/annotations/instances_val2017.json
```

# TODO
1. This framework is continuously being modified, and it is currently a header-only source, which might substantially increase the compilation time when it grows.
2. Add a proper logging and stats collection module.
