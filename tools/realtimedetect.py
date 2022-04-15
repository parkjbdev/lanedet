import numpy as np
import torch
import cv2
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes as drawLanes
from lanedet.utils.net_utils import load_network
from tools import draw

class ObjectDetection(object):
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5x6", pretrained=True)
        self.object_of_interest = np.array(
            [0, 1, 2, 3, 5, 7, 9, 10, 11, 12], dtype=np.uint8
        )

    def getObjects(self, frame):
        result = self.model(frame)
        objects = result.pandas().xyxy[0].iloc
        return objects

    def draw(self, frame, objects):
        for obj in objects:
            name, confidence = obj["name"], obj.confidence
            xmin, xmax = int(obj.xmin), int(obj.xmax)
            ymin, ymax = int(obj.ymin), int(obj.ymax)
            if not obj["class"] in self.object_of_interest:
                continue

            draw.boundingBox(frame, xmin, ymin, xmax, ymax, name, confidence)


class LaneDetection(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids=range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        img = ori_img[self.cfg.cut_height :, :, :].astype(np.float32)
        data = {"img": img, "lanes": []}
        data = self.processes(data)
        data["img"] = data["img"].unsqueeze(0)
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
            lanes = [lane.to_array(self.cfg) for lane in data[0]]

        return lanes

    def draw(self, frame, lanes):
        drawLanes(frame, lanes)

    def getLanes(self, frame):
        data = self.preprocess(frame)
        return self.inference(data)


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/condlane/resnet101_culane.py',help='The path of config file')
parser.add_argument('--source', default='drive.mp4', help='The source of the video')
parser.add_argument('--load_from', type=str, default='weights/condlane_r101_culane.pth', help='The path of model')
parser.add_argument('--start_from', type=int, default='1300', help='Start Video from specific second')
args = parser.parse_args()

cap = cv2.VideoCapture(args.source)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_MSEC, args.start_from * 1000)

print(f"{WIDTH}x{HEIGHT} starting from {args.start_from} second")

# LaneDetection Configs
cfg = Config.fromfile(args.config)
cfg.load_from = args.load_from
laneDetection = LaneDetection(cfg)

# Yolov5 Configs
yolo = ObjectDetection()

while True:
    ret, frame = cap.read()

    objects = yolo.getObjects(frame)
    yolo.draw(frame, objects)

    lanes = laneDetection.getLanes(frame)
    laneDetection.draw(frame, lanes)

    draw.fpsmeter(frame)
    cv2.imshow("frame", frame)

    cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()