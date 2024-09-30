import cv2

def main():
    # Web kamerasını başlat
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera açılamadı")
        return

    while True:
        # Kameradan görüntü oku
        ret, frame = cap.read()

        if not ret:
            print("Görüntü alınamadı")
            break

        # Görüntüyü ekranda göster
        cv2.imshow("Web Kamerası", frame)

        # Esc tuşuna basıldığında çıkış yap
        if cv2.waitKey(1) & 0xFF == 27:  # 27, Esc tuşunun ASCII değeridir
            print("Çıkış yapılıyor...")
            break

    # Kamerayı ve pencereleri serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
