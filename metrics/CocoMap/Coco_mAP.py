import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pylab
import argparse

# pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm', 'bbox', 'keypoints']
annType = annType[1]  # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))

# initialize COCO ground truth api
currentDir = os.getcwd()
dataDir = currentDir + '/results/'

cocoGt = None


def parse_opts():
    parser = argparse.ArgumentParser(description="Coco mAP")
    parser.add_argument('--result_file', type=str, default="")
    parser.add_argument('--result_dir', type=str, default="")
    parser.add_argument('--annotation_file', type=str)
    args = parser.parse_args()

    return args


def compute_mAP(resFile):
    print("mAP for file: ", resFile)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    args = parse_opts()

    cocoGt = COCO(args.annotation_file)
    if args.result_dir != "":
        for resFile in sorted(os.listdir(args.result_dir)):
            compute_mAP(args.result_dir + "/" + resFile)

    if args.result_file != "":
        compute_mAP(args.result_file)
