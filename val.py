# -*- coding: utf-8 -*-
# @Time : 2024/3/18 20:08
# @Author : Weiqi
# @File : val.py

# -*- coding: utf-8 -*-
# @Time : 2024/3/12 18:04
# @Author : Weiqi
# @File : train.py.py

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("F:\\processing\\3.12\\yolov8\\runs\\detect\\train12\\weights\\best.pt")
    model = YOLO("F:\\processing\\3.12\\yolov8\\runs\\detect\\train12\\weights\\best.pt")

    model.val(model="F:\\processing\\3.12\\yolov8\\runs\\detect\\train12\\weights\\best.pt",
              data="F:\\processing\\3.12\\yolov8\\ultralytics\\cfg\\datasets\\my_data.yaml", epochs=300, imgsz=640)
