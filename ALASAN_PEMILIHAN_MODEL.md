# Alasan Pemilihan 6 Model Machine Learning

Dokumen ini menjelaskan alasan mengapa 6 model Machine Learning ini dipilih dalam project, serta kelebihan dan kekurangan masing-masing. Penjelasan dibuat dengan bahasa sederhana agar mudah dipahami.

---

## 1. K-Means Clustering (Unsupervised — Tradisional)

### Alasan Dipilih
- **Sederhana dan cepat** — algoritma clustering yang paling dasar dan mudah dipahami
- **Mudah diinterpretasikan** — hasil segmentasi mudah dijelaskan di laporan
- **Standar industri** — hampir semua project clustering menggunakan K-Means sebagai baseline
- **Efisien untuk data besar** — bisa memproses ribuan data dengan cepat

### Kelebihan
- Implementasi sangat mudah
- Waktu komputasi cepat
- Cocok sebagai model pembanding dasar (baseline)

### Kekurangan
- Hanya bisa membentuk cluster berbentuk bulat (spherical)
- Hasil sangat dipengaruhi oleh inisialisasi awal centroid
- Tidak bisa menangani outlier dengan baik
- Jumlah cluster (k) harus ditentukan sebelum training

### Kapan Cocok Dipakai
Ketika ingin melihat gambaran awal segmentasi data secara cepat dan sederhana.

---

## 2. Gaussian Mixture Model / GMM (Unsupervised — Modern)

### Alasan Dipilih
- **Soft clustering** — memberikan probabilitas keanggotaan, bukan kelompok tegas. Ini lebih realistis karena seorang developer bisa saja memiliki karakteristik dari beberapa kelompok sekaligus
- **Bentuk cluster fleksibel** — tidak terbatas bentuk bulat seperti K-Means, bisa berbentuk ellipsoid
- **Perbandingan modern vs tradisional** — tujuan project membandingkan pendekatan tradisional (K-Means) dengan modern (GMM)

### Kelebihan
- Memberikan informasi probabilitas (seberapa yakin suatu data masuk ke cluster tertentu)
- Cluster bisa memiliki berbagai bentuk dan ukuran
- Lebih akurat untuk data yang kompleks

### Kekurangan
- Lebih lambat dari K-Means
- Butuh lebih banyak data untuk hasil yang stabil
- Rentan terhadap inisialisasi yang buruk

### Kapan Cocok Dipakai
Ketika data memiliki struktur yang kompleks dan tidak bulat sempurna, atau ketika ingin tahu seberapa yakin suatu data masuk ke suatu kelompok.

---

## 3. Random Forest (Supervised — Tradisional)

### Alasan Dipilih
- **Model ensemble yang kuat** — menggabungkan banyak pohon keputusan sehingga hasilnya lebih stabil
- **Handal untuk data tabular** — sangat cocok untuk dataset seperti survei yang didominasi data kategorikal
- **Menampilkan feature importance** — bisa diketahui fitur mana yang paling berpengaruh dalam prediksi
- **Tidak mudah overfitting** — karena menggunakan rata-rata dari banyak pohon
- **Standar baseline** — Random Forest adalah model klasifikasi standar yang selalu dipakai sebagai pembanding

### Kelebihan
- Bisa menangani data numerik dan kategorikal sekaligus
- Stabil terhadap noise dan outlier
- Tidak perlu scaling fitur

### Kekurangan
- Ukuran model bisa sangat besar (banyak pohon)
- Butuh lebih banyak memori
- Tidak sebaik neural network untuk pola non-linear yang sangat kompleks

### Kapan Cocok Dipakai
Ketika ingin model yang stabil, interpretatif, dan andal sebagai baseline klasifikasi.

---

## 4. Linear SVM / Support Vector Machine (Supervised — Tradisional)

### Alasan Dipilih
- **Model linear sederhana** — sebagai pembanding untuk melihat apakah data bisa dipisahkan secara linear
- **Efektif untuk dimensi tinggi** — dataset setelah preprocessing memiliki 379 fitur, SVM bekerja baik di ruang berdimensi tinggi
- **Margin maksimal** — SVM mencari garis pemisah dengan jarak terbesar antar kelas, sehingga generalisasinya lebih baik
- **Melengkapi perbandingan tradisional** — bersama Random Forest, mewakili kelompok model tradisional

### Kelebihan
- Efektif untuk data dengan banyak fitur
- Secara teori memiliki generalisasi yang baik
- Cukup cepat untuk training

### Kekurangan
- Hanya linear — tidak bisa menangkap pola non-linear tanpa kernel trick
- Performa paling rendah di antara semua model (F1 = 0,760)
- Butuh scaling fitur

### Kapan Cocok Dipakai
Ketika ingin baseline model linear dan menguji apakah data dapat dipisahkan dengan garis lurus.

---

## 5. XGBoost (Supervised — Modern)

### Alasan Dipilih
- **Boosting modern terpopuler** — XGBoost adalah algoritma yang mendominasi kompetisi data science dan industri
- **Performa tinggi untuk data tabular** — sering menjadi pemenang di kompetisi Kaggle untuk dataset tabular
- **Menangani imbalance data** — parameter `scale_pos_weight` bisa diatur untuk menangani ketidakseimbangan kelas (78% : 22%)
- **Perbandingan modern vs tradisional** — mewakili kelompok model modern bersama MLP

### Kelebihan
- Akurasi sangat tinggi untuk data tabular
- Menangani missing value secara otomatis
- Regularisasi internal untuk mencegah overfitting
- Fitur importance untuk interpretasi

### Kekurangan
- Butuh tuning parameter yang lebih teliti
- Rentan overfitting jika tidak diatur dengan baik
- Waktu training lebih lama dari Random Forest

### Kapan Cocok Dipakai
Ketika ingin performa tinggi dan dataset didominasi fitur tabular/kategorikal.

---

## 6. MLP Neural Network (Supervised — Modern)

### Alasan Dipilih
- **Mewakili deep learning** — menguji apakah neural network bisa mengungguli model klasik pada dataset tabular
- **Menangkap pola non-linear kompleks** — dengan hidden layer, MLP bisa mempelajari hubungan yang sangat kompleks
- **Perbandingan tradisional vs modern** — melengkapi perbandingan dengan Random Forest dan SVM (tradisional)
- **Fleksibel** — arsitektur bisa disesuaikan (jumlah layer, neuron, aktivasi)

### Kelebihan
- **Performa terbaik** — F1-score tertinggi (0,882) di antara semua model
- Mampu menangkap pola non-linear yang tidak bisa ditangkap model linear
- Arsitektur bisa dikustomisasi

### Kekurangan
- Butuh lebih banyak data untuk training yang stabil
- Waktu training lebih lama
- Kurang interpretatif (black box) — sulit menjelaskan mengapa model mengambil keputusan
- Butuh tuning hyperparameter

### Kapan Cocok Dipakai
Ketika fokus utama adalah performa prediksi tertinggi, bukan interpretasi.

---

## Tabel Ringkasan Alasan Pemilihan

| No | Model | Kategori | Alasan Utama Dipilih | F1-Score |
|----|-------|----------|---------------------|----------|
| 1 | **K-Means** | Unsupervised Tradisional | Baseline clustering yang sederhana dan cepat | — |
| 2 | **GMM** | Unsupervised Modern | Soft clustering yang lebih fleksibel | — |
| 3 | **Random Forest** | Supervised Tradisional | Ensemble stabil, interpretatif, baseline kuat | 0,867 |
| 4 | **Linear SVM** | Supervised Tradisional | Baseline model linear, uji keterpisahan data | 0,760 |
| 5 | **XGBoost** | Supervised Modern | Boosting populer, performa tinggi untuk tabular | 0,789 |
| 6 | **MLP** | Supervised Modern | Neural network, menangkap pola kompleks | **0,882** |

---

## Pola Pemilihan Model

Project ini menggunakan pola **perbandingan berpasangan**:

```
Unsupervised:   K-Means (Tradisional)   vs   GMM (Modern)
Supervised:     Random Forest + SVM     vs   XGBoost + MLP
                (Tradisional)                 (Modern)
```

Tujuannya adalah untuk membandingkan apakah pendekatan modern benar-benar lebih unggul dari pendekatan tradisional dalam konteks dataset survei developer ini.
