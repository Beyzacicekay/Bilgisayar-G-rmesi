from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # YOLOv8 nano modeli kullanılıyor

# Test görüntüsü yükle
img = cv2.imread('yoloo.jpg')  # test_image.jpg dosyasını proje klasörüne ekle

# Modeli görüntü üzerinde çalıştır
results = model(img)

# Sonuçları görselleştir
for result in results:
    # Sonuçları göstermek için çizim yapıyoruz
    plt.imshow(result.plot())  # result.plot() ile sonucun üzerine çizim yapılır
    plt.show()

# Sonuçları kaydet
for i, result in enumerate(results):
    result.save(f"outputs/result_{i}.jpg")
