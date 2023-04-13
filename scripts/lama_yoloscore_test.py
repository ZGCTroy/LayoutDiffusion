from PIL import Image
import os
import numpy as np
import argparse
import json
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from cocoapi.PythonAPI.pycocotools import mask as maskUtils
import re


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def getImageID():
    pass


def main(args):
    getImageID()
    # image_ID = np.loadtxt(args.imageid_path)
    # image_ID = image_ID.astype(int)
    os.makedirs('image_result', exist_ok=True)
    file_list = os.listdir(args.image_path)

    print("resizing pic....")
    for i in range(0, len(file_list)):
        img = Image.open(os.path.join(args.image_path, file_list[i]))
        out2 = img.resize((512, 512))
        out2.save(os.path.join('image_result', file_list[i]))

    print("writing into txt..")
    with open('train.txt', 'w') as fp:
        for i in range(len(file_list)):
            fp.writelines(os.path.join('../data/image_result/', file_list[i]) + '\n')

    print("using yolov4....")
    commands = 'cd ../darknet;./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out ../data/json_result/result.json < ../data/train.txt'
    os.system(commands)

    image_ID = []

    for filename in file_list:
        image_ID.append(int(re.findall(r"(\d+)", filename)[0]))

    print('transforming.....')
    with open('json_result/result.json', 'r', encoding='utf8') as fp:
        yolo_data = json.load(fp)
    coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                        77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                        82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                        88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

    coco_id = {value: key for key, value in coco_id_name_map.items()}
    result = []
    elements = {}
    for i in range(0, len(yolo_data)):
        image_id = image_ID[i]
        image_id = str(image_id).zfill(12)
        im = Image.open(os.path.join("val2017", f"{image_id}.jpg"))
        # im = Image.open()
        (orgin_width, orgin_height) = im.size
        for j in range(0, len(yolo_data[i]['objects'])):
            elements["image_id"] = image_ID[i]
            if yolo_data[i]['objects'][j]['name'] == 'diningtable':
                elements["category_id"] = coco_id['dining table']
            elif yolo_data[i]['objects'][j]['name'] == 'tvmonitor':
                elements["category_id"] = coco_id['tv']
            elif yolo_data[i]['objects'][j]['name'] == 'pottedplant':
                elements["category_id"] = coco_id['potted plant']
            elif yolo_data[i]['objects'][j]['name'] == 'aeroplane':
                elements["category_id"] = coco_id['airplane']
            elif yolo_data[i]['objects'][j]['name'] == 'motorbike':
                elements["category_id"] = coco_id['motorcycle']
            elif yolo_data[i]['objects'][j]['name'] == 'sofa':
                elements["category_id"] = coco_id['couch']
            else:
                elements["category_id"] = coco_id[yolo_data[i]['objects'][j]['name']]

            temp_x = yolo_data[i]['objects'][j]['relative_coordinates']['center_x'] - 0.5 * \
                     yolo_data[i]['objects'][j]['relative_coordinates']['width']
            temp_y = yolo_data[i]['objects'][j]['relative_coordinates']['center_y'] - 0.5 * \
                     yolo_data[i]['objects'][j]['relative_coordinates']['height']
            temp_w = yolo_data[i]['objects'][j]['relative_coordinates']['width']
            temp_h = yolo_data[i]['objects'][j]['relative_coordinates']['height']
            elements["bbox"] = [temp_x * orgin_width,
                                temp_y * orgin_height,
                                temp_w * orgin_width,
                                temp_h * orgin_height]
            elements["score"] = round(yolo_data[i]['objects'][j]['confidence'], 3)
            # print(elements)
            result.append(elements.copy())
    with open('json_result/result_final.json', 'w') as f:
        json.dump(result, f, cls=MyEncoder)

    print("eval on filtered annotations.....")
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]
    cocoGt = COCO("instances_val2017_filtered.json")
    cocoDt = cocoGt.loadRes("json_result/result_final.json")
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = image_ID
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("Done!")

    print("eval on full annotations.....")
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]
    cocoGt = COCO("instances_val2017.json")
    cocoDt = cocoGt.loadRes("json_result/result_final.json")
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = image_ID
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageid_path', type=str, default='image_id.txt', help='get image id')
    parser.add_argument('--image_path', type=str, default='128/', help='which epoch to load')
    args = parser.parse_args()
    main(args)
