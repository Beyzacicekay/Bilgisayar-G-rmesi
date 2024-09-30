import cv2
import numpy as np

def apply_filters(image_path): #bir görüntü dosya yolu argüman olarak alır
    # Görüntüyü oku
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Görüntü yüklenemedi.")
        return

    # Görüntüyü gri tonlamaya çevir
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) #görüntüyü renkli formattan gri tonlama

    # Ortalama filtre uygula (blurring)
    average_filtered_image = cv2.blur(gray_image, (5, 5))

    # Laplace filtresi uygula
    laplace_filtered_image = cv2.Laplacian(average_filtered_image, cv2.CV_64F)
    laplace_filtered_image = np.uint8(np.absolute(laplace_filtered_image))  # Negatif değerleri pozitif yap görüntüyü 8 bitlik sayı dizisine çevir

    # Görüntüleri yanyana birleştir
    combined_image = np.hstack((gray_image, average_filtered_image, laplace_filtered_image))

    # Görüntüleri göster
    cv2.imshow('Orijinal - Ortalama Filtre - Laplace Filtresi', combined_image)

    # Çıkış için bir tuşa basılmasını bekle
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Görüntü yolunu burada belirt
    image_path = 'lena2.jpg'  # Buraya görüntü dosyanızın yolunu girin
    apply_filters(image_path)
