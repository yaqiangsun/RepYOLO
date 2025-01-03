# -*- coding: utf-8 -*-
# @Time : 2024/3/18 20:16
# @Author : Weiqi
# @File : predict.py

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("F:\\processing\\3.12\\yolov8\\runs\\detect\\train31\\weights\\best.pt")
    model = YOLO("F:\\processing\\3.12\\yolov8\\runs\\detect\\train31\\weights\\best.pt")

    model.predict(model="F:\\processing\\3.12\\yolov8\\runs\\detect\\train31\\weights\\best.pt",
                  source="F:\\processing\\3.12\\yolov8\\test_data", save=True)
