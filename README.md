# 🚗 Toyota Used Car Price Prediction (Machine Learning End-to-End Pipeline)

Proyek Machine Learning untuk memprediksi harga mobil bekas (*Used Car Price Prediction*) merk **Toyota** berdasarkan spesifikasi kendaraan dan riwayat penggunaan. Proyek ini mengimplementasikan alur kerja data science dan machine learning yang lengkap mulai dari pengumpulan data hingga pembangunan model prediktif.

---

## 📌 Daftar Isi
- [Ringkasan Proyek](#-ringkasan-proyek)
- [Struktur Dataset](#-struktur-dataset)
- [Alur Kerja (Machine Learning Pipeline)](#-alur-kerja-machine-learning-pipeline)
  - [1. Data Collecting](#1-mengumpulkan-data-data-collecting)
  - [2. Data Loading](#2-data-loading)
  - [3. Data Cleaning & Transformation](#3-data-cleaning--transformation)
  - [4. Exploratory Data Analysis (EDA & ExDA)](#4-eda--exda)
  - [5. Data Splitting](#5-data-splitting)
  - [6. Pembangunan Model (Modelling)](#6-pembangunan-model-modelling)
- [Hasil dan Evaluasi Model](#-hasil-dan-evaluasi-model)
- [Struktur Repositori](#-struktur-repositori)
- [Panduan Menjalankan Proyek](#-panduan-menjalankan-proyek)

---

## 📖 Ringkasan Proyek
Penentuan harga pasar untuk mobil bekas sering kali dipengaruhi oleh kombinasi berbagai faktor teknis seperti tahun pembuatan, jarak tempuh (*mileage*), jenis transmisi, kapasitas mesin, dan efisiensi bahan bakar (*mpg*). 

Tujuan dari proyek ini adalah:
1. Membangun model regresi yang mampu memprediksi nilai harga jual mobil (`price`) secara akurat.
2. Membandingkan performa beberapa algoritma regresi (*Least Angle Regression*, *Linear Regression*, dan *Gradient Boosting Regressor*).
3. Mengekspor model terbaik untuk kebutuhan deployment atau inferensi lebih lanjut.

---

## 📊 Struktur Dataset
Dataset yang digunakan adalah **Toyota Used Car Dataset** (`toyota.csv`) yang terdiri dari **6.738 baris data** dan **9 kolom fitur**.

🔗 **Link Unduh Dataset (Kaggle):**  
👉 [Used Car Dataset - Ford and Mercedes (Kaggle)](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes) *(Gunakan file `toyota.csv` di dalam dataset tersebut)*

| Nama Kolom | Tipe Data | Deskripsi |
|---|---|---|
| `model` | Kategorikal | Tipe/model mobil Toyota (Yaris, Corolla, Aygo, C-HR, RAV4, dll.) |
| `year` | Numerik (Integer) | Tahun produksi / registrasi mobil |
| `transmission` | Kategorikal | Jenis sistem transmisi (Manual, Automatic, Semi-Auto, Other) |
| `mileage` | Numerik (Integer) | Total jarak yang telah ditempuh (dalam mil) |
| `fuelType` | Kategorikal | Jenis bahan bakar (Petrol, Diesel, Hybrid, Other) |
| `tax` | Numerik (Integer) | Pajak jalan tahunan (dalam GBP) |
| `mpg` | Numerik (Float) | Konsumsi bahan bakar (*Miles Per Gallon*) |
| `engineSize` | Numerik (Float) | Kapasitas volume mesin kendaraan (dalam liter) |
| **`price`** | Numerik (Integer) | **Target Variable**: Harga jual mobil (dalam GBP) |

---

## 🛠 Alur Kerja (Machine Learning Pipeline)

Tahapan pada notebook `Praktek.ipynb` disusun secara modular dan terstruktur:

### 1. Mengumpulkan Data (Data Collecting)
- Pembahasan prinsip pengumpulan data dari sumber terbuka (*Open Data Sources*) seperti Kaggle, UCI Machine Learning Repository, Google Dataset Search Engine, dan portal data pemerintah.
- Penentuan rumusan masalah, identifikasi fitur independen ($X$), serta penentuan target label ($y$).
- Verifikasi ketersediaan file `toyota.csv` yang diunduh dari [Kaggle Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes).

### 2. Data Loading
- Membaca dan memuat dataset CSV ke dalam memori menggunakan library **Pandas** (`pd.read_csv`).
- Pemeriksaan ringkasan awal data dengan fungsi `.head()` dan pengecekan dimensi matriks data.

### 3. Data Cleaning & Transformation
- **Analisis Missing Values**: Pengecekan nilai kosong dan penanganan menggunakan metode imputasi (median untuk fitur numerik dan modus/mode untuk fitur kategorikal).
- **Deteksi & Penanganan Outlier**: Menggunakan metode *Interquartile Range* (IQR) untuk memfilter nilai ekstrem yang dapat mendistorsi model, dilengkapi visualisasi boxplot.
- **Feature Scaling**: Melakukan standardisasi fitur numerik menggunakan `StandardScaler` agar distribusi data memiliki skala yang seimbang (mean = 0, variance = 1).
- **Deduplikasi**: Pemeriksaan dan penghapusan baris duplikat.
- **Encoding Variabel Kategorikal**: Mengonversi fitur bertipe objek ke representasi numerik menggunakan `LabelEncoder` dan One-Hot Encoding (`pd.get_dummies`).

### 4. EDA & ExDA (Exploratory Data Analysis)
- Visualisasi distribusi frekuensi setiap variabel menggunakan grid histogram subplot.
- Analisis distribusi probabilitas pada variabel numerik utama (`year`, `mileage`, `tax`, `mpg`, `engineSize`).
- Analisis multivariat menggunakan *Correlation Matrix Heatmap*.
- Analisis korelasi langsung antara setiap fitur prediktor terhadap target harga (`price`).

### 5. Data Splitting
- Pemisahan matriks prediktor ($X$) dan target ($y$).
- Pembagian data menggunakan `train_test_split` dari scikit-learn dengan rasio **80% data latih (training set)** dan **20% data uji (testing set)** serta `random_state=1` untuk memastikan hasil dapat direproduksi (*reproducibility*).

### 6. Pembangunan Model (Modelling)
Melatih 3 algoritma regresi:
1. **Least Angle Regression (LARS)**
2. **Linear Regression (LR)**
3. **Gradient Boosting Regressor (GBR)**

Evaluasi dilakukan pada *testing set* menggunakan 3 metrik standar regresi:
- **MAE** (*Mean Absolute Error*)
- **MSE** (*Mean Squared Error*)
- **$R^2$ Score** (*Coefficient of Determination*)

---

## 📈 Hasil dan Evaluasi Model

Perbandingan performa model pada data pengujian:

| Algoritma | MAE | MSE | $R^2$ Score | Catatan |
|---|:---:|:---:|:---:|---|
| **Least Angle Regression (LARS)** | 0.5708 | 0.5643 | 0.4415 | Kurang mampu menangkap pola non-linear |
| **Linear Regression (LR)** | 0.3536 | 0.2370 | 0.7654 | Performa baseline yang cukup baik |
| **Gradient Boosting Regressor (GBR)** 🏆 | **0.1537** | **0.0465** | **0.9540** | **Performa terbaik** ($R^2$ mencapai **95,4%**) |

Model terbaik (**Gradient Boosting Regressor**) diekspor dan disimpan ke dalam file:
- `gbr_model.joblib`
- `gbr_model.pkl`

---

## 📁 Struktur Repositori

```text
├── Praktek.ipynb        # Jupyter Notebook lengkap (Collecting s/d Modelling)
├── toyota.csv           # Dataset mobil bekas Toyota
├── gbr_model.joblib     # Model Gradient Boosting tersimpan (format Joblib)
├── gbr_model.pkl        # Model Gradient Boosting tersimpan (format Pickle)
└── README.md            # Dokumentasi lengkap proyek
```

---

## 🚀 Panduan Menjalankan Proyek

### 1. Clone Repositori
```bash
git clone https://github.com/Willkycmz/<nama-repo>.git
cd <nama-repo>
```

### 2. Buat & Aktifkan Virtual Environment (Opsional tapi Disarankan)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
```

### 4. Jalankan Notebook
Buka menggunakan VS Code atau Jupyter Lab / Notebook:
```bash
jupyter notebook Praktek.ipynb
```

---

## 👨‍💻 Kontributor
- **William Cemas** ([@Willkycmz](https://github.com/Willkycmz))
