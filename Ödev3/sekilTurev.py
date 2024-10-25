#Author Beyza
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüleri yükle (renkli)
kare_img = cv2.imread("karee.jpg")
daire_img = cv2.imread("dairee.jpg")

# Gri tonlamaya dönüştür
kare_gray = cv2.cvtColor(kare_img, cv2.COLOR_BGR2GRAY)
daire_gray = cv2.cvtColor(daire_img, cv2.COLOR_BGR2GRAY)

# Sobel filtreleri ile türev hesapla (kenarları daha iyi yakalar)
# Yatay türev (x yönünde)
sobelx_kare = cv2.Sobel(kare_gray, cv2.CV_64F, 1, 0, ksize=3)
sobelx_daire = cv2.Sobel(daire_gray, cv2.CV_64F, 1, 0, ksize=3)

# Dikey türev (y yönünde)
sobely_kare = cv2.Sobel(kare_gray, cv2.CV_64F, 0, 1, ksize=3)
sobely_daire = cv2.Sobel(daire_gray, cv2.CV_64F, 0, 1, ksize=3)

# Sonuçları normalize etmek yerine mutlak değerlere dönüştürelim
sobelx_kare = cv2.convertScaleAbs(sobelx_kare)
sobelx_daire = cv2.convertScaleAbs(sobelx_daire)
sobely_kare = cv2.convertScaleAbs(sobely_kare)
sobely_daire = cv2.convertScaleAbs(sobely_daire)

# Sonuçları görselleştir
fig, axs = plt.subplots(2, 4, figsize=(15, 8))

# Orijinal görüntüler (renkli)
axs[0, 0].imshow(cv2.cvtColor(kare_img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Orijinal Kare")
axs[1, 0].imshow(cv2.cvtColor(daire_img, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title("Orijinal Daire")

# Yatay ve dikey türev sonuçlarını gri tonlamalı gösterimi
axs[0, 1].imshow(sobelx_kare, cmap='gray')
axs[0, 1].set_title("Yatay Türev - Kare")
axs[0, 2].imshow(sobely_kare, cmap='gray')
axs[0, 2].set_title("Dikey Türev - Kare")

axs[1, 1].imshow(sobelx_daire, cmap='gray')
axs[1, 1].set_title("Yatay Türev - Daire")
axs[1, 2].imshow(sobely_daire, cmap='gray')
axs[1, 2].set_title("Dikey Türev - Daire")

# Ekseni kapat
for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
