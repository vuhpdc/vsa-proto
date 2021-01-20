import cv2 as cv
import argparse
import json
import time
import pandas as pd


def parse_opts():
    parser = argparse.ArgumentParser(
        description="Visualize video and bounding boxes")

    parser.add_argument("--video_file", type=str,
                        help='Path to video file')
    parser.add_argument("--annotation_gt_file", type=str,
                        help='Path to annotation gt file')
    parser.add_argument("--annotation_det_file", type=str,
                        help='Path to annotation det file')
    parser.add_argument("--label_file", type=str,
                        help="Path to COCO label file")

    args = parser.parse_args()
    args_dict = args.__dict__
    print("------------------------------------")
    print("Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")

    return args


def readLabel(file):
    # df = pd.read_csv(file, delimiter=",", header=None, names=["Label"])
    # return df["Label"].tolist()

    with open(file, 'r') as coco:
        js = json.loads(coco.read())
        categories = js['categories']
    total = categories[-1]["id"]
    labels = []
    j = 0
    for i in range(total+1):
        if (i == categories[j]["id"]):
            labels.append(categories[j]["name"])
            j += 1
        else:
            labels.append("N/A")
    return labels


def readAnnotation(input_file):
    output = {}
    # read lines from the input json file
    with open(input_file) as json_data:
        input_lines = json.load(json_data)
        for input_line in input_lines:
            image_id = input_line['image_id']
            if image_id not in output:
                output[image_id] = []
            output_line = [input_line['category_id'],
                           input_line['bbox'], input_line['score']]
            output[image_id].append(output_line)

    return output


def show(opts):
    cap = cv.VideoCapture(opts.video_file)
    if not(cap.isOpened()):
        print("Cannot open ", opts.video_file)
        return

    annotations_gt = readAnnotation(opts.annotation_gt_file)
    annotations_det = readAnnotation(opts.annotation_det_file)
    labels = readLabel(opts.label_file)

    frame_id = 0
    while (True):
        ret, frame = cap.read()
        frame_gt_annotations = annotations_gt[frame_id]
        frame_det_annotations = annotations_det[frame_id]
        for i in range(len(frame_gt_annotations)):
            # Ground truth annotations
            annotation = frame_gt_annotations[i]
            label = int(annotation[0])
            boxes = annotation[1]
            p1 = (int(boxes[0]), int(boxes[1]))
            p2 = (int(p1[0]+boxes[2]), int(p1[1]+boxes[3]))

            cv.rectangle(frame, p1, p2, (0, 255, 0), 1)

            (_, _), baseline = cv.getTextSize(labels[label],
                                              cv.FONT_HERSHEY_SIMPLEX,
                                              0.75, 1)
            cv.putText(frame, labels[label], (p1[0], p1[1] - baseline),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        # Detection annotations
        for i in range(len(frame_det_annotations)):
            annotation = frame_det_annotations[i]
            label = int(annotation[0])
            boxes = annotation[1]
            p1 = (int(boxes[0]), int(boxes[1]))
            p2 = (int(p1[0]+boxes[2]), int(p1[1]+boxes[3]))

            cv.rectangle(frame, p1, p2, (255, 0, 0), 1)

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1
        time.sleep(30 * 1e-3)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    opts = parse_opts()

    show(opts)
