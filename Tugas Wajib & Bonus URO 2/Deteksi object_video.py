import cv2
import numpy as np # numpy digunakan untuk bekerja dengan array

# Membaca video
video = cv2.VideoCapture(r"C:\Users\ASUS\OneDrive\Dokumen\Matkul ITB\URO\Tugas URO 2\object_video.mp4") 
# membuka file dengan r di awal untuk menunjukkan raw string agar ketika \ dimasukkan, sistem tidak error

if not video.isOpened(): # jika video tidak terbuka, sistem akan mengeluarkan output error
    print("Error: Video tidak terbuka.")
    exit() # Tujuannya agar user tidak menunggu lama video tidak terbuka - buka

# Mengatur rentang warna yang ingin dideteksi dalam ruang warna HSV
# Karena bola berwarna merah, gunakan rgb batas bawah dan atas warna merah
lower_color = np.array([0, 100, 100])
upper_color = np.array([179, 255, 255])
# Rentang warna dalam HSV, Hue (warna), Saturation (kejenuhan), dan Value (kecerahan)

while True:
    ret, frame = video.read()
    if not ret:
        break
# video.read() digunakan untuk membaca setiap frame dalam video
# Jika ret bernilai False, video telah selesai digunakan, dan loop berhenti

    frame = cv2.resize(frame, (640, 480)) # mengubah ukuran frame 640 x 480 pixel agar tidak terlalu besar
    
    # Mengonversi frame dari RGB ke HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masking menggunakan rentang warna
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    # cv2.inRange membuat mask dengan nilai 1 untuk piksel yang ada dalam 
    # - rentang warna yang telah ditentukan dan nilai 0 untuk yang tidak

    # Menghilangkan noise dari mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # erode untuk mengurangi ukuran objek, sering digunakan untuk menghilangkan noise (piksel kecil yang tidak diinginkan)
    # dilate memperbesar objek, digunakan untuk menambahkan detail ke objek atau untuk menyatukan bagian-bagian objek yang terpisah.


    # Menemukan kontur dari objek yang di-mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # findContours digunakan untuk mencari kontur (garis yang membentuk objek dalam gambar) dalam mask
    # cv2.RETR_EXTERNAL hanya mencari kontur eksternal (bukan kontur dalam objek)
    # cv2.CHAIN_APPROX_SIMPLE digunakan untuk mengurangi jumlah titik dalam kontur (penyederhanaan)

    # Menggambar bounding rectangle di sekitar objek yang memenuhi kriteria
    for contour in contours:
        area = cv2.contourArea(contour) # mengukur luas area kontur
        if area > 200:  # Objek harus lebih besar dari 200
            x, y, w, h = cv2.boundingRect(contour) # menghitung koordinat dan ukuran bounding rectangle untuk objek tersebut
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2) # menggambar rectangle di sekitar objek yang terdeteksi

    def close_window(): # function untuk mematikan semua programnya ketika sudah selesai
        video.release() # mematikan kamera dan menutup semua OpenCV windows yang terbuka
        cv2.destroyAllWindows()
        exit()

    # Menampilkan frame dengan bounding rectangle
    cv2.imshow('Deteksi Bola Merah', frame) 
    # parameter pertama untuk nama windows yang akan terbuka
    # parameter kedua untuk memproses frame 

    # Tekan 'q' untuk keluar dari program
    if cv2.waitKey(30) & 0xFF == ord('q'):
        close_window() # memangil function close_window untuk melepaskan video dan menutup semua window
