#Author Beyza
import pandas as pd
import requests
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")

# CSV dosyasını oku
df = pd.read_csv('goruntulink.csv')

# Sonuçları depolamak için bir liste
results = []

for index, row in df.iterrows():
    image_url = row['image_url']

    try:
        # Resmi indir
        response = requests.get(image_url, stream=True, verify=False, timeout=5)
        response.raise_for_status()

        image_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)

        # YOLO ile tespit et
        results_yolo = model(image)

        # Tespit edilen nesneler
        objects = []

        for res in results_yolo[0].boxes:
            class_name = model.names[int(res.cls[0])]
            x1, y1, x2, y2 = map(int, res.xyxy[0])

            # Nesneyi kare içine al
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Metin kutusunu arka planla beraber çiz
            text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_w, text_h = text_size

            # Metin kutusunu yerleştir (karenin üstünde veya yanında)
            if y1 - text_h - 10 < 0:  # Eğer yukarıda boşluk yoksa
                y1 = text_h + 10  # Metni biraz daha aşağıda göstermek için ayarla

            # Arka plan kutusu ekle (metnin görünürlüğünü artırmak için)
            cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)

            # Metni görüntüye yaz
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            # Nesneyi listeye ekle
            objects.append(class_name)

        found_person = 'person' in objects
        results.append({
            'image_url': image_url,
            'contains_person': found_person,
            'detected_objects': ', '.join(objects)
        })

        # Görüntüyü orantılı bir şekilde yeniden boyutlandır
        height, width = image.shape[:2]
        scale_factor = 500 / height
        resized_image = cv2.resize(image, (int(width * scale_factor), 500))

        cv2.imshow(f"Image {index + 1}", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except requests.exceptions.RequestException as e:
        print(f"Could not retrieve image from {image_url}. Error: {e}")

# Sonuçları CSV dosyasına kaydet
output_df = pd.DataFrame(results)
output_df.to_csv('yolo_sonuclar.csv', index=False)

print("İşlem tamamlandı, sonuçlar 'yolo_sonuclar.csv' dosyasına kaydedildi.")
