import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8-C2f-UIB-CA-Adown.yaml')
    model.train(data=r'/root/yolov8/ultralytics/cfg/datasets/fruit.yaml',
 
                cache=False,
                imgsz=640,
                epochs=200,
                single_cls=False,  
                batch=64,
                close_mosaic=0,
                workers=4,
                device='0',
                optimizer='SGD', 
 
                amp=True,  
                project='runs/train',
                name='exp',
                )
