# -*- coding: utf-8 -*-
# @Time : 2024/3/12 18:04
# @Author : Weiqi
# @File : train.py.py

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("F:\\processing\\3.12\\yolov8\\yolov8l.pt")
    model = YOLO("F:\\processing\\3.12\\yolov8\\yolov8.yaml")

    model.train(model="F:\\processing\\3.12\\yolov8\\yolov8l.pt",
                data="F:\\processing\\3.12\\yolov8\\ultralytics\\cfg\\datasets\\my_data.yaml", epochs=300, patience=200,
                imgsz=640)
