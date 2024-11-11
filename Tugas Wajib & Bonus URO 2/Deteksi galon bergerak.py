import cv2
import numpy as np

# Membaca video dari video file
video = cv2.VideoCapture(r"C:\Users\ASUS\OneDrive\Dokumen\Matkul ITB\URO\Tugas URO 2\Galon bergerak.mp4")  

# Definisikan rentang warna untuk mendeteksi galon warna biru
lower_color = np.array([100, 50, 50])  # Rentang warna bawah (BGR)
upper_color = np.array([140, 255, 255])  # Rentang warna atas (BGR)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Mengubah frame menjadi ruang warna HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Menyaring warna berdasarkan rentang yang telah ditentukan
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Menggunakan filter untuk mengurangi noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # morphologyEx mirip dengan fungsi erode dan dilate (digabung)
    # (np.ones((5, 5), np.uint8) membuat array 5x5 yang berisi angka 1, dengan tipe data unsigned 8-bit integer

    # Menemukan kontur pada gambar
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Menyaring kontur kecil (hanya objek besar yang terdeteksi)
            continue

        # Menggambar bounding box di sekitar objek yang terdeteksi
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def close_window(): # function untuk mematikan semua programnya ketika sudah selesai
        video.release() 
        cv2.destroyAllWindows()
        exit()

    # Menampilkan hasil deteksi
    cv2.imshow('Deteksi Galon Bergerak', frame) # Nama windownya Deteksi Galon Bergerak

    # Keluar jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        close_window() # memanggil function close_window untuk melepaskan video dan menutup semua window

