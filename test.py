import cv2
from ultralytics import YOLO
if __name__ == '__main__':
    model=YOLO(model="yolov8n.yaml")
    model.load("runs/detect/train4/weights/last.pt")
    model.train(data="dataset.yaml", epochs=20, imgsz=640,device=0,resume=True)
