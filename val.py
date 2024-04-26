from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO(model="runs/detect/train5/weights/best.pt")
    
    # 指定测试数据的路径
    test_data_path = "datasets/Vehicles/test\images"
        
    metrics = model.val(data='dataset.yaml')
    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)


