import numpy as np
import torch
import cv2
import argparse
from lanedet.datasets.process import Process
from lanedet.models.registry import build_net
from lanedet.utils.config import Config
from lanedet.utils.visualization import imshow_lanes
from lanedet.utils.net_utils import load_network

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, ori_img):
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img':ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.get_lanes(data)
        return data

    def show(self, data):
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=True)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        self.show(data)
        return data


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

# Configs
cfg = Config.fromfile(args.config)
cfg.load_from = args.load_from

detect = Detect(cfg)

while True:
    ret, frame = cap.read()
    data = detect.run(frame)

    cv2.waitKey(1)

cap.release()