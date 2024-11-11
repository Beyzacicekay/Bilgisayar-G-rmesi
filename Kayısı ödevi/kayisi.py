

import torch
from ultralytics import YOLO
from roboflow import Roboflow


print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli


rf = Roboflow(api_key="y2t7hjKFDkmss3rLw33n")
project = rf.workspace("bilgisayar-grmesi").project("kayisi-rkg3g")
version = project.version(1)
dataset = version.download("yolov8")

# Eğitim ve test işlemlerini ana kod bloğuna alıyoruz
# Modeli eğit
if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=100)