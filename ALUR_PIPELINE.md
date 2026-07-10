# Alur Pipeline Machine Learning — Stack Overflow AI Tools

Dokumen ini menjelaskan alur kerja (pipeline) project Machine Learning ini dari awal sampai akhir dengan bahasa yang mudah dipahami, tanpa istilah teknis yang rumit.

---

## 1. Gambaran Besar Project

Project ini bertujuan untuk **menganalisis dan memprediksi penggunaan AI Tools oleh para developer** (programmer) di seluruh dunia. Data yang digunakan adalah hasil survei Stack Overflow tahun 2025.

Ada dua tujuan utama:

| Tujuan | Metode | Penjelasan Sederhana |
|--------|--------|----------------------|
| **Segmentasi** (Unsupervised Learning) | K-Means & GMM | Mengelompokkan developer ke dalam beberapa kelompok berdasarkan kesamaan karakteristik mereka |
| **Prediksi** (Supervised Learning) | Random Forest, SVM, XGBoost, MLP | Memprediksi apakah seorang developer menggunakan AI Tools atau tidak, berdasarkan data diri dan teknologi yang mereka gunakan |

---

## 2. Dataset yang Digunakan

Dataset berasal dari **Stack Overflow Developer Survey 2025**, yaitu survei tahunan yang diisi oleh ribuan developer dari seluruh dunia. Survei ini berisi banyak pertanyaan seperti:

- Berapa usia responden?
- Dari negara mana?
- Bahasa pemrograman apa yang dikuasai?
- Database apa yang pernah digunakan?
- Berapa lama pengalaman coding?
- Apakah menggunakan AI Tools? (jawaban ini dijadikan target/tebakan)

Dari total **49.191 responden**, setelah dibersihkan, data yang layak pakai sebanyak **33.720 responden**.

---

## 3. Alur Pipeline — Dari Awal sampai Akhir

Berikut adalah diagram alur kerja secara berurutan:

```
LOAD DATASET
     ↓
PREPROCESSING (Bersihkan Data)
     ↓
     ├──→ UNSUPERVISED LEARNING (Segmentasi)
     │       ├── K-Means Clustering
     │       └── GMM (Gaussian Mixture Model)
     │
     └──→ SUPERVISED LEARNING (Prediksi)
             ├── Train/Test Split
             ├── Random Forest
             ├── Linear SVM
             ├── XGBoost
             └── MLP Neural Network
                    ↓
             Evaluasi Semua Model
                    ↓
             Simpan Hasil (Tabel + Grafik)
```

---

## 4. Tahap 1 — Preprocessing (Persiapan Data)

Sebelum data bisa diproses oleh model, data mentah harus dibersihkan dan diubah dulu. Tahap ini dilakukan oleh file `src/data_preprocessing.py`.

### Langkah-langkahnya:

**a. Load Dataset**
Membaca file CSV survei menggunakan pandas.

**b. Membuat Target (Label)**
Kolom `AISelect` berisi jawaban "Yes" atau "No" tentang penggunaan AI. Ini diubah menjadi:
- `1` = menggunakan AI Tools
- `0` = tidak menggunakan AI Tools

**c. Memilih Fitur yang Relevan**
Tidak semua kolom dipakai. Hanya 13 kolom kandidat yang dipilih, misalnya: Age, Country, EdLevel, DevType, YearsCode, LanguageHaveWorkedWith, dll.

**d. Menghindari Data Leakage (Kebocoran Data)**
Kolom yang terlalu dekat dengan target (misalnya kolom tentang AI lainnya) TIDAK boleh dipakai sebagai fitur. Tujuannya agar model benar-benar belajar dari karakteristik developer, bukan dari jawaban AI yang sudah bocor.

**e. Membersihkan Data**
- Missing value: diisi nilai tengah (median) untuk angka, atau "Unknown" untuk kategori
- Kolom YearsCode: nilai seperti "Less than 1 year" diubah jadi 0, "More than 50 years" diubah jadi 51
- Fitur multi-select (kolom yang bisa diisi lebih dari satu jawaban, dipisah titik koma) diubah menjadi format biner

**f. Transformasi Fitur**
- Fitur numerik (YearsCode) → distandarisasi dengan StandardScaler
- Fitur kategorikal (Country, Employment) → diubah ke One-Hot Encoding
- Fitur multi-select (LanguageHaveWorkedWith, dll) → diubah dengan CountVectorizer

Hasil akhir: dari 11 fitur asli, preprocessing menghasilkan **379 fitur** siap pakai.

---

## 5. Tahap 2 — Unsupervised Learning (Segmentasi Developer)

Tahap ini bertujuan **mengelompokkan developer** ke dalam kelompok-kelompok (cluster) yang memiliki karakteristik mirip. Tidak ada label/ jawaban benar — model mencari pola sendiri.

### 5.1 K-Means Clustering

**Cara kerja sederhana:**
1. Tentukan jumlah kelompok (k)
2. Ambil k titik acak sebagai pusat kelompok (centroid)
3. Hitung jarak setiap data ke setiap centroid
4. Kelompokkan data ke centroid terdekat
5. Hitung ulang posisi centroid berdasarkan rata-rata data di kelompoknya
6. Ulangi sampai posisi centroid tidak berubah

**Hasil:**
- Jumlah kelompok terbaik: **k = 2**
- **Cluster 0 (55,8% developer)**: developer dengan pengalaman lebih banyak, dominan Python
- **Cluster 1 (44,2% developer)**: developer dengan pengalaman lebih sedikit, dominan JavaScript/TypeScript

### 5.2 GMM (Gaussian Mixture Model)

**Perbedaan dengan K-Means:**
- K-Means: setiap developer masuk ke SATU kelompok secara tegas (hard clustering)
- GMM: setiap developer punya PROBABILITAS untuk masuk ke setiap kelompok (soft clustering) — lebih fleksibel

**Cara kerja sederhana:**
1. Asumsikan data berasal dari beberapa distribusi normal (bentuk lonceng)
2. Hitung probabilitas setiap data terhadap setiap kelompok
3. Perbarui parameter distribusi (mean, covariance, weight)
4. Ulangi sampai stabil

**Hasil:**
- Jumlah kelompok terbaik: **k = 3**
- Cluster 0 (42,86%): pengalaman rata-rata 9,55 tahun
- Cluster 1 (14,10%): pengalaman rata-rata 16,29 tahun
- Cluster 2 (43,05%): pengalaman rata-rata 25,06 tahun

---

## 6. Tahap 3 — Supervised Learning (Prediksi Penggunaan AI)

Tahap ini bertujuan **memprediksi apakah seorang developer menggunakan AI Tools atau tidak**, berdasarkan fitur-fitur yang sudah disiapkan.

Data dibagi dua:
- **80% data training**: untuk melatih model
- **20% data testing**: untuk menguji performa model

### 6.1 Random Forest (Tradisional)

**Cara kerja sederhana:**
- Membuat banyak pohon keputusan (decision tree) dari data yang diacak
- Setiap pohon memberikan tebakan sendiri
- Hasil akhir diambil dari voting terbanyak

**Parameter:**
- 200 pohon (n_estimators=200)
- class_weight = balanced (menangani data yang tidak seimbang)

**Hasil:** F1-score = **0,867**

### 6.2 Linear SVM (Tradisional)

**Cara kerja sederhana:**
- Mencari garis pemisah (hyperplane) terbaik antara dua kelas
- Garis dipilih yang memberikan jarak terbesar antara kelas

**Parameter:**
- C = 1.0
- max_iter = 5000

**Hasil:** F1-score = **0,760** (paling rendah)

### 6.3 XGBoost (Modern)

**Cara kerja sederhana:**
- Membangun model secara berurutan (sequential)
- Setiap model baru fokus memperbaiki kesalahan model sebelumnya
- Menggunakan teknik gradient boosting

**Parameter:**
- 200 pohon, learning_rate = 0,1, max_depth = 6

**Hasil:** F1-score = **0,789**

### 6.4 MLP Neural Network (Modern)

**Cara kerja sederhana:**
- Meniru cara kerja otak manusia dengan neuron buatan
- Terdiri dari: input layer → 2 hidden layer (100 neuron, 50 neuron) → output layer
- Menggunakan aktivasi ReLU dan Adam optimizer
- Early stopping untuk mencegah overfitting

**Hasil:** F1-score = **0,882** (**TERBAIK**)

---

## 7. Evaluasi Model

Setelah semua model dilatih, performanya diukur dengan metrik berikut:

| Metrik | Penjelasan Sederhana |
|--------|---------------------|
| **Accuracy** | Persentase tebakan yang benar dari total tebakan |
| **Precision** | Dari semua yang ditebak "pakai AI", berapa yang benar-benar pakai? |
| **Recall** | Dari semua yang benar-benar pakai AI, berapa yang berhasil ditebak? |
| **F1-Score** | Rata-rata harmonis antara Precision dan Recall |

### Hasil Perbandingan (diurutkan dari terbaik)

| Model | F1-Score | Akurasi |
|-------|----------|---------|
| **MLP Neural Network** | **0,882** | 0,792 |
| Random Forest | 0,867 | 0,858 |
| XGBoost | 0,789 | 0,751 |
| Linear SVM | 0,760 | 0,768 |

---

## 8. Output yang Dihasilkan

Semua hasil disimpan di folder `outputs/`:

| Folder | Isi |
|--------|-----|
| `outputs/figures/` | Grafik-grafik: elbow method, silhouette score, confusion matrix, perbandingan model, distribusi cluster |
| `outputs/tables/` | Tabel-tabel CSV: metrik model, confusion matrix, classification report, ringkasan cluster |
| `outputs/reports/` | Ringkasan hasil untuk laporan BAB IV |

---

## 9. Ringkasan Alur dalam Satu Paragraf

Data survei Stack Overflow 2025 dibersihkan dan disiapkan → fitur-fitur penting dipilih (usia, negara, bahasa, pengalaman, dll) → dua pendekatan dilakukan secara paralel: **(1) Segmentasi** menggunakan K-Means dan GMM untuk mengelompokkan developer, dan **(2) Prediksi** menggunakan Random Forest, SVM, XGBoost, dan MLP untuk menebak apakah developer memakai AI Tools → hasilnya dibandingkan → model MLP menjadi yang terbaik → semua grafik dan tabel disimpan untuk bahan laporan.

---

## 10. Cara Menjalankan Pipeline

Ada dua cara:

**Via Terminal:**
```bash
python src/run_pipeline.py
```

**Via Jupyter Notebook:**
```bash
jupyter notebook
# Lalu buka: notebooks/01_model_stackoverflow_ai_tools.ipynb
```

Keduanya akan menjalankan alur yang sama persis dari awal sampai akhir.
