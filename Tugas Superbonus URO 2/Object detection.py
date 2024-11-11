import cv2
from ultralytics import YOLO

# Memuat model YOLO
yolo = YOLO('yolov8s.pt')

# Memuat gambar
image_path = "Keramaian.jpg"  # Path ke gambar Anda
image = cv2.imread(image_path)

# Memeriksa apakah gambar berhasil dimuat
if image is None:
    print("Error, gambar tidak ditemukan.")
else:
    # Fungsi untuk mendapatkan warna berdasarkan nomor kelas
    def getColours(cls_num):
        base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        color_index = cls_num % len(base_colors)
        increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
        color = [base_colors[color_index][i] + increments[color_index][i] *
                 (cls_num // len(base_colors)) % 256 for i in range(3)]
        return tuple(color)

    # Menjalankan deteksi pada gambar
    results = yolo.predict(image)

    if not results: # memeriksa apakah ada objek yang ingin diamati
        print("Tidak ada deteksi ditemukan.")
    else:
        print("Deteksi ditemukan:", results)

    # Mengambil hasil deteksi dan menggambar kotak pada gambar
    for result in results:
        classes_names = result.names  # Nama kelas
        for box in result.boxes:
            # Memeriksa jika confidence lebih dari 20%
            if box.conf[0] > 0.2: # confidence sengaja diset tidak terlalu tinggi karena gambar memang cukup padat
                # Mengambil koordinat kotak deteksi
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Mendapatkan kelas dan nama kelas
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = getColours(cls)

                # Menggambar kotak deteksi dan menampilkan label kelas
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(image, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                
                print(f'Terdeteksi: {class_name} di [{x1}, {y1}, {x2}, {y2}] dengan confidence {box.conf[0]:.2f}')
    # Menampilkan hasil deteksi
    cv2.imshow('Deteksi pada Gambar', image)
    cv2.waitKey(0)  # Tekan sembarang tombol untuk menutup jendela
    cv2.destroyAllWindows()
