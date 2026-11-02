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
└── README.md            # Dokumentasi lengkap proyek
```

---

## 🚀 Panduan Menjalankan Proyek

### 1. Clone Repositori
```bash
git clone https://github.com/Willkycmz/toyota-used-car-price-prediction-with-Regession.git
cd toyota-used-car-price-prediction-with-Regession
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


# Prediksi Harga Mobil Bekas Toyota Menggunakan Regresi (Machine Learning)

Proyek ini membangun dan membandingkan model *machine learning* untuk memprediksi harga mobil bekas Toyota (`price`) berdasarkan spesifikasi kendaraan dan riwayat penggunaannya. Algoritma yang diuji adalah **Least Angle Regression (LARS)**, **Linear Regression**, dan **Gradient Boosting Regressor (GBR)**. Kinerja model diukur menggunakan **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, dan **Koefisien Determinasi (R²)**.

> **Catatan sitasi:** Sitasi seperti `[1, hlm. 194–195]` mengarah ke nomor referensi pada Daftar Pustaka dan halaman artikel jurnal yang tercetak pada PDF. Isi yang digunakan dalam landasan teori ditulis ulang sebagai parafrase, bukan salinan langsung. Rincian bagian, halaman, dan persamaan yang menjadi dasar kutipan tersedia pada **Bagian 8 – Peta Sumber Kutipan**.

## Daftar Isi

1. [Ringkasan Proyek](#1-ringkasan-proyek)
2. [Dataset](#2-dataset)
3. [Alur Kerja](#3-alur-kerja)
4. [Landasan Teori](#4-landasan-teori)
   - [4.1 Machine Learning dan Regresi](#41-machine-learning-dan-regresi)
   - [4.2 Prediksi Harga Mobil Bekas](#42-prediksi-harga-mobil-bekas)
   - [4.3 Preprocessing, EDA, dan Pembagian Data](#43-preprocessing-eda-dan-pembagian-data)
   - [4.4 Algoritma Regresi](#44-algoritma-regresi)
   - [4.5 Metrik Evaluasi Model](#45-metrik-evaluasi-model)
5. [Penelitian Terkait](#5-penelitian-terkait)
6. [Hasil dan Evaluasi Model](#6-hasil-dan-evaluasi-model)
7. [Pembahasan dan Keterbatasan](#7-pembahasan-dan-keterbatasan)
8. [Peta Sumber Kutipan](#8-peta-sumber-kutipan)
9. [Daftar Pustaka](#9-daftar-pustaka)
10. [Struktur Repositori](#10-struktur-repositori)
11. [Cara Menjalankan Proyek](#11-cara-menjalankan-proyek)
12. [Kontributor](#12-kontributor)

---

## 1. Ringkasan Proyek

Harga mobil bekas tidak ditentukan oleh satu faktor saja. Tahun kendaraan, jarak tempuh, tipe atau model, jenis transmisi, bahan bakar, pajak kendaraan, konsumsi bahan bakar, dan kapasitas mesin dapat menjadi informasi yang relevan dalam memperkirakan harga. Penelitian tentang prediksi harga mobil bekas Toyota juga menjelaskan bahwa penentuan harga yang tidak sesuai dapat merugikan penjual maupun pembeli [1, hlm. 194–195].

Tujuan proyek ini adalah:

1. Membangun model regresi untuk memperkirakan harga jual mobil bekas Toyota.
2. Membandingkan hasil tiga algoritma, yaitu LARS, Linear Regression, dan Gradient Boosting Regressor.
3. Mengevaluasi hasil prediksi menggunakan MAE, MSE, dan R².
4. Menyimpan model GBR untuk digunakan kembali pada proses prediksi.

## 2. Dataset

Dataset yang digunakan adalah `toyota.csv`, yang merupakan bagian dari *Used Car Dataset – Ford and Mercedes* di Kaggle. Berdasarkan pemeriksaan pada notebook proyek, dataset Toyota terdiri dari **6.738 baris dan 9 kolom**.

**Sumber dataset:** [Used Car Dataset – Ford and Mercedes (Kaggle)](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes). Gunakan berkas `toyota.csv` dari dataset tersebut.

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `model` | Kategorikal | Model mobil Toyota, misalnya Yaris, Corolla, Aygo, C-HR, dan RAV4 |
| `year` | Numerik | Tahun produksi/registrasi kendaraan |
| `transmission` | Kategorikal | Jenis transmisi, seperti Manual, Automatic, dan Semi-Auto |
| `mileage` | Numerik | Jarak tempuh kendaraan dalam mil |
| `fuelType` | Kategorikal | Jenis bahan bakar, seperti Petrol, Diesel, dan Hybrid |
| `tax` | Numerik | Pajak jalan tahunan dalam GBP |
| `mpg` | Numerik | Konsumsi bahan bakar dalam *miles per gallon* |
| `engineSize` | Numerik | Kapasitas mesin dalam liter |
| `price` | Numerik | **Target** yang diprediksi, yaitu harga jual dalam GBP |

## 3. Alur Kerja

Tahapan yang dijalankan pada notebook `Praktek.ipynb` dirangkum berikut. Kolom terakhir menunjukkan hubungan tahap implementasi dengan landasan teori pada Bagian 4.

| Tahap | Implementasi pada notebook | Landasan teori |
|---|---|---|
| Data collecting | Menggunakan `toyota.csv` dari Kaggle; menetapkan fitur dan target | 4.1 dan 4.2 |
| Data loading | Memuat data menggunakan Pandas | 4.3 |
| Missing value | Imputasi median untuk numerik dan modus untuk kategorikal; kolom dengan jumlah missing value yang sangat banyak dihapus | 4.3 |
| Outlier | Boxplot dan metode IQR | 4.3 |
| Scaling | `StandardScaler` pada kolom numerik | 4.3 |
| Encoding | `LabelEncoder`; `pd.get_dummies` juga dibuat sebagai hasil transformasi terpisah | 4.3 |
| EDA | Histogram dan *correlation matrix heatmap* | 4.3 |
| Data splitting | `train_test_split`, 80% data latih dan 20% data uji, `random_state=1` | 4.3 |
| Modelling | LARS, Linear Regression, dan Gradient Boosting Regressor | 4.4 |
| Evaluasi | MAE, MSE, dan R² pada data uji | 4.5 |

## 4. Landasan Teori

### 4.1 Machine Learning dan Regresi

*Machine learning* merupakan pendekatan komputasi yang memungkinkan sistem mempelajari pola dari data untuk menghasilkan prediksi atau keputusan tanpa seluruh aturan harus ditentukan secara manual. Pada pembelajaran terawasi (*supervised learning*), proses pelatihan menggunakan data yang mempunyai pasangan fitur masukan dan target keluaran. Model mempelajari hubungan dari pasangan tersebut, kemudian menggunakan pola yang diperoleh untuk memperkirakan target pada data baru. Pendekatan regresi digunakan ketika target yang ingin diprediksi berbentuk nilai numerik atau kontinu, misalnya harga, suhu, jumlah produksi, atau nilai ekonomi [2, hlm. 1305; 3, hlm. 1126–1127].

Dalam proyek ini, fitur kendaraan menjadi variabel independen atau prediktor, sedangkan `price` menjadi variabel dependen atau target. Karena harga merupakan nilai numerik, masalah ini termasuk **regresi**, bukan klasifikasi. Perbedaan tersebut berpengaruh terhadap cara mengevaluasi model. Evaluasi klasifikasi umumnya menggunakan akurasi, *precision*, *recall*, atau F1-score, sedangkan evaluasi regresi dapat menggunakan MAE, MSE, dan R² [2, hlm. 1305].

Secara umum, model regresi berusaha menghasilkan nilai prediksi yang mendekati nilai aktual. Selisih antara nilai aktual dan prediksi disebut residual atau galat prediksi. Evaluasi dilakukan dengan mengukur besar galat tersebut dan menilai seberapa baik model menjelaskan variasi target. Oleh karena itu, penilaian model tidak cukup hanya melihat satu angka; beberapa metrik dapat digunakan karena masing-masing memberikan sudut pandang yang berbeda terhadap kesalahan prediksi [2, hlm. 1305; 3, hlm. 1127–1128].

### 4.2 Prediksi Harga Mobil Bekas

Prediksi harga mobil bekas merupakan permasalahan regresi karena nilai yang ingin diperkirakan berupa harga numerik. Penentuan harga dapat dipengaruhi oleh beberapa karakteristik kendaraan, seperti model, tahun pembuatan, jarak tempuh, jenis transmisi, tipe bahan bakar, pajak, konsumsi bahan bakar, dan kapasitas mesin. Studi yang secara khusus membahas prediksi harga mobil bekas Toyota juga menggunakan karakteristik kendaraan untuk membangun model prediksi harga [1, hlm. 194–196].

Informasi tahun dan jarak tempuh dapat membantu merepresentasikan usia serta intensitas pemakaian kendaraan. Sementara itu, model, jenis transmisi, bahan bakar, dan kapasitas mesin menggambarkan karakteristik kendaraan yang dapat membedakan satu unit dengan unit lainnya. Hubungan antara karakteristik tersebut dan harga tidak harus selalu berbentuk garis lurus. Karena itu, proyek ini membandingkan algoritma regresi linier dengan algoritma berbasis *boosting* yang dapat mempelajari hubungan lebih kompleks [1, hlm. 195; 4, hlm. 388–389].

Prediksi yang dihasilkan model merupakan estimasi berdasarkan pola pada dataset, bukan penilaian kondisi fisik kendaraan secara langsung. Dataset proyek tidak mencakup seluruh faktor yang mungkin memengaruhi harga, misalnya kondisi bodi, riwayat kecelakaan, kelengkapan dokumen, lokasi penjualan, atau kondisi pasar pada saat transaksi. Dengan demikian, hasil model perlu dipahami sebagai estimasi berdasarkan fitur yang tersedia, bukan sebagai harga jual pasti.

### 4.3 Preprocessing, EDA, dan Pembagian Data

Sebelum digunakan untuk pelatihan, data perlu diperiksa dan disiapkan agar formatnya sesuai dengan kebutuhan algoritma. Tahap ini meliputi pemeriksaan nilai kosong, duplikasi, pencilan, transformasi fitur kategorikal, penyetaraan skala fitur, dan eksplorasi hubungan antarvariabel. Studi prediksi harga mobil bekas Toyota juga menerapkan eksplorasi dan transformasi data sebelum melatih model regresi [1, hlm. 196–198].

**a. Missing value dan data duplikat**

*Missing value* merupakan kondisi ketika suatu kolom tidak memiliki nilai pada baris tertentu. Nilai kosong dapat ditangani dengan menghapus baris/kolom yang tidak layak digunakan atau mengisi nilai menggunakan pendekatan imputasi. Pemilihan metode imputasi perlu mempertimbangkan jenis data dan sebaran nilainya. Median dapat digunakan untuk fitur numerik karena tidak terlalu dipengaruhi nilai ekstrem, sedangkan modus dapat digunakan untuk fitur kategorikal karena merupakan kategori yang paling sering muncul. Penghapusan data duplikat dilakukan untuk menghindari pencatatan observasi yang sama lebih dari satu kali. Penelitian terdahulu juga menjelaskan pemeriksaan missing value dan duplikasi sebagai bagian dari persiapan dataset [1, hlm. 196; 4, hlm. 385].

Pada notebook proyek, nilai kosong pada fitur numerik diisi dengan median, sedangkan fitur kategorikal diisi dengan modus. Kolom yang memiliki setidaknya 1.000 nilai kosong dihapus dari data yang diproses. Setelah itu, baris duplikat diidentifikasi dan dihapus.

**b. Outlier dan Interquartile Range (IQR)**

*Outlier* adalah observasi yang nilainya jauh berbeda dari sebagian besar data. Keberadaan pencilan dapat memengaruhi proses pembentukan model, terutama ketika model sensitif terhadap nilai yang sangat jauh dari pola umum. Salah satu cara untuk mendeteksinya adalah menggunakan *Interquartile Range* (IQR), yaitu selisih antara kuartil ketiga (Q3) dan kuartil pertama (Q1). Batas bawah dan batas atas yang umum digunakan adalah [1, hlm. 197]:

\[
IQR = Q3 - Q1
\]

\[
Batas\ bawah = Q1 - 1{,}5 \times IQR
\]

\[
Batas\ atas = Q3 + 1{,}5 \times IQR
\]

Data yang berada di luar batas tersebut dapat ditandai sebagai pencilan untuk ditinjau lebih lanjut. Pada notebook proyek, boxplot digunakan untuk membantu pemeriksaan dan baris yang terindikasi sebagai outlier pada fitur numerik disaring menggunakan batas IQR.

**c. Encoding variabel kategorikal**

Algoritma regresi pada umumnya memerlukan masukan berbentuk numerik. Oleh karena itu, fitur kategorikal seperti `model`, `transmission`, dan `fuelType` perlu diubah ke representasi angka. *Label Encoding* memberikan kode bilangan pada setiap kategori. Sementara itu, *One-Hot Encoding* membuat kolom biner terpisah untuk masing-masing kategori sehingga kategori tidak langsung diperlakukan sebagai tingkatan angka [5, hlm. 3481–3482].

Pada jalur pemodelan notebook ini, data akhir yang digunakan untuk membentuk `X` dan `y` berasal dari dataframe hasil **LabelEncoder** (`df_lencoder`). Notebook juga membuat dataframe One-Hot Encoding (`df_one_hot`), tetapi dataframe tersebut bukan dataframe yang digunakan untuk melatih ketiga model pada bagian akhir notebook.

**d. Standardisasi fitur**

Standardisasi mengubah skala data numerik agar fitur memiliki skala yang sebanding. Salah satu tekniknya adalah *Z-score standardization*, yang mengubah nilai berdasarkan rata-rata dan standar deviasi fitur [4, hlm. 386; 5, hlm. 3482]. Rumusnya adalah:

\[
z = \frac{x - \mu}{\sigma}
\]

Keterangan:
- \(x\) = nilai asli suatu fitur.
- \(\mu\) = rata-rata fitur.
- \(\sigma\) = standar deviasi fitur.
- \(z\) = nilai setelah standardisasi.

Pada notebook digunakan `StandardScaler` dari scikit-learn. Perlu diperhatikan bahwa implementasi saat ini menerapkan scaler pada seluruh kolom numerik yang terpilih, termasuk kolom target `price`, dan proses `fit_transform` dilakukan sebelum pembagian data latih dan data uji. Dampaknya terhadap interpretasi nilai MAE dan MSE dijelaskan kembali pada Bagian 7.

**e. Exploratory Data Analysis (EDA)**

*Exploratory Data Analysis* (EDA) merupakan tahap eksplorasi untuk memahami struktur, distribusi, dan hubungan antarvariabel sebelum pemodelan. EDA dapat dilakukan melalui statistik deskriptif, histogram, boxplot, dan matriks korelasi. Histogram membantu melihat distribusi masing-masing fitur, sedangkan *correlation matrix* dapat memberikan gambaran awal mengenai arah dan kekuatan hubungan linear antarfitur. Hasil eksplorasi dapat digunakan sebagai bahan untuk memahami karakteristik dataset dan menentukan hal-hal yang perlu diperhatikan pada tahap pemodelan [4, hlm. 386; 5, hlm. 3484–3485].

Korelasi tidak otomatis membuktikan hubungan sebab-akibat dan tidak selalu cukup untuk menentukan penting tidaknya suatu fitur bagi semua jenis model. Karena itu, hasil EDA digunakan sebagai analisis awal, bukan satu-satunya dasar untuk menyimpulkan pengaruh suatu variabel.

**f. Pembagian data latih dan data uji**

Dataset umumnya dibagi menjadi data latih (*training set*) dan data uji (*testing set*). Data latih digunakan untuk mempelajari pola serta parameter model, sedangkan data uji digunakan untuk mengukur performa pada observasi yang tidak dipakai dalam pelatihan. Pemisahan ini diperlukan agar evaluasi memberikan gambaran kemampuan model menghadapi data yang belum dilihat [2, hlm. 1305; 5, hlm. 3482–3483].

Proyek ini menggunakan pembagian 80% data latih dan 20% data uji melalui `train_test_split(test_size=0.2, random_state=1)`. Nilai `random_state` ditetapkan agar pembagian dapat direproduksi. Selain satu kali pembagian, *cross-validation* dapat digunakan pada pengembangan berikutnya untuk mengevaluasi model melalui beberapa lipatan data dan mengurangi ketergantungan terhadap satu pembagian tertentu [5, hlm. 3483].

### 4.4 Algoritma Regresi

Tiga algoritma regresi digunakan untuk membandingkan pendekatan linier dan pendekatan ensemble berbasis pohon keputusan.

**a. Linear Regression**

*Linear Regression* merupakan metode statistik yang memodelkan hubungan antara variabel dependen dan satu atau lebih variabel independen menggunakan bentuk hubungan linier. Pada regresi linier berganda, prediksi target dibentuk dari konstanta dan penjumlahan kontribusi setiap fitur. Model berusaha menentukan koefisien yang membuat nilai prediksi mendekati nilai aktual pada data pelatihan [3, hlm. 1126–1127; 5, hlm. 3482].

Persamaan regresi linier berganda dapat dituliskan sebagai:

\[
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p
\]

Keterangan:
- \(\hat{y}\) = nilai target hasil prediksi.
- \(\beta_0\) = konstanta atau *intercept*.
- \(\beta_j\) = koefisien regresi untuk fitur ke-j.
- \(x_j\) = nilai fitur independen ke-j.
- \(p\) = jumlah fitur prediktor.

Kelebihan regresi linier adalah struktur model relatif sederhana dan koefisiennya dapat ditafsirkan. Namun, model ini memiliki keterbatasan ketika pola hubungan antara fitur dan target sangat nonlinier, karena bentuk dasarnya mengasumsikan hubungan linier. Dalam proyek ini, Linear Regression digunakan sebagai salah satu model pembanding atau *baseline*.

**b. Least Angle Regression (LARS)**

*Least Angle Regression* (LARS) merupakan algoritma regresi yang membangun model secara bertahap dengan memilih prediktor berdasarkan hubungan atau korelasinya terhadap residual. Pada awal proses, belum ada prediktor aktif. Algoritma kemudian memilih prediktor yang paling berkaitan dengan residual, menggerakkan koefisien pada arah yang seimbang terhadap prediktor aktif, dan menambahkan prediktor lain ketika memenuhi kondisi pemilihan berikutnya. Proses tersebut membentuk jalur koefisien secara bertahap dan dapat digunakan dalam konteks pemilihan variabel [6, bagian “Annex 3: Least-Angle Regression (LARS) Algorithm”].

Berbeda dari Gradient Boosting yang menggabungkan pohon keputusan, LARS menghasilkan model regresi linier. Oleh karena itu, kemampuannya tetap berkaitan dengan representasi linier fitur terhadap target. Pada notebook proyek, LARS dibuat menggunakan `Lars(n_nonzero_coefs=1)`. Parameter tersebut membatasi jumlah koefisien tidak nol yang digunakan model, sehingga konfigurasi yang diuji merupakan model LARS dengan jumlah prediktor aktif yang sangat terbatas.

**c. Gradient Boosting Regressor (GBR)**

*Gradient Boosting Regressor* merupakan metode ensemble yang membangun model secara bertahap dengan menggabungkan beberapa *weak learner*, yang pada implementasi umum berupa pohon keputusan. Pada setiap iterasi, model menambahkan pohon baru untuk memperbaiki kesalahan atau residual dari model sebelumnya. Dengan proses bertahap tersebut, Gradient Boosting dapat mempelajari pola yang lebih kompleks dan hubungan nonlinier dalam data [4, hlm. 386 dan 388–389].

Secara konseptual, pembentukan model aditif dapat digambarkan sebagai:

\[
F_m(x) = F_{m-1}(x) + \nu h_m(x)
\]

Keterangan:
- \(F_m(x)\) = prediksi model setelah iterasi ke-m.
- \(F_{m-1}(x)\) = model pada iterasi sebelumnya.
- \(h_m(x)\) = kontribusi pohon baru pada iterasi ke-m.
- \(\nu\) = *learning rate* yang mengatur besar kontribusi pohon baru.

Persamaan tersebut merupakan ilustrasi konseptual proses penambahan model, bukan salinan persamaan dari artikel yang dirujuk. *Learning rate*, jumlah pohon, dan kedalaman pohon merupakan contoh parameter yang dapat memengaruhi performa serta kompleksitas model. Pada notebook proyek, model dibuat menggunakan `GradientBoostingRegressor(random_state=184)` dengan parameter selain `random_state` mengikuti nilai bawaan scikit-learn.

### 4.5 Metrik Evaluasi Model

Evaluasi dilakukan untuk mengetahui seberapa dekat hasil prediksi terhadap nilai aktual dan seberapa baik model menjelaskan variasi target. MAE, MSE, dan R² merupakan metrik yang digunakan dalam evaluasi regresi [2, hlm. 1305; 3, hlm. 1127–1128]. Ketiganya memiliki cara interpretasi yang berbeda, sehingga hasilnya perlu dibaca secara bersamaan.

Notasi yang digunakan:
- \(n\) = jumlah observasi pada data uji.
- \(y_i\) = nilai aktual pada observasi ke-i.
- \(\hat{y}_i\) = nilai prediksi model pada observasi ke-i.
- \(\bar{y}\) = rata-rata nilai aktual.

**a. Mean Absolute Error (MAE)**

*Mean Absolute Error* merupakan rata-rata nilai absolut dari selisih antara nilai aktual dan nilai prediksi. Nilai absolut membuat kesalahan positif dan negatif tidak saling menghapus. MAE menunjukkan besarnya kesalahan prediksi rata-rata dalam satuan yang sama dengan target yang dievaluasi [1, hlm. 198; 3, hlm. 1127].

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

Nilai MAE yang semakin kecil menunjukkan rata-rata selisih absolut prediksi semakin kecil. Sebagai contoh, apabila target dievaluasi dalam GBP, MAE sebesar 500 berarti rata-rata selisih absolut prediksi sekitar 500 GBP. Namun, interpretasi dalam satuan GBP hanya berlaku ketika target dan prediksi memang masih berada pada skala GBP. Pada notebook proyek ini, `price` ikut distandardisasi, sehingga MAE yang tercantum pada hasil saat ini bukan dalam GBP.

**b. Mean Squared Error (MSE)**

*Mean Squared Error* merupakan rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi. Proses pengkuadratan menyebabkan kesalahan yang besar mendapat penalti lebih besar dibandingkan kesalahan yang kecil. Dengan demikian, MSE berguna untuk melihat besarnya kesalahan dengan penekanan lebih kuat pada prediksi yang meleset jauh [1, hlm. 198; 3, hlm. 1128].

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

Nilai MSE yang lebih kecil menunjukkan galat kuadrat rata-rata yang lebih kecil. Karena selisih dikuadratkan, satuan MSE merupakan kuadrat satuan target. Apabila target menggunakan GBP, satuan MSE secara konseptual adalah GBP². Hal ini membuat nilai MSE tidak dibaca sebagai selisih harga langsung seperti MAE.

**c. Koefisien Determinasi (R²)**

Koefisien determinasi atau R² mengukur proporsi variasi nilai target yang dapat dijelaskan oleh model dibandingkan dengan pendekatan dasar yang memprediksi rata-rata target. R² dihitung dari perbandingan jumlah kuadrat residual (*sum of squared residuals*) dengan jumlah kuadrat total (*total sum of squares*) [3, hlm. 1128].

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

Dengan:

\[
SS_{res} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

\[
SS_{tot} = \sum_{i=1}^{n}(y_i - \bar{y})^2
\]

Nilai R² terbaik adalah 1. Nilai 0 menunjukkan bahwa model tidak lebih baik daripada pendekatan sederhana yang selalu memprediksi rata-rata target pada data evaluasi. **R² juga dapat bernilai negatif** apabila prediksi model lebih buruk daripada pendekatan rata-rata tersebut. Karena itu, R² tidak tepat dijelaskan sebagai metrik yang selalu berada pada rentang 0 sampai 1; dokumentasi scikit-learn menyatakan bahwa skor R² dapat bernilai negatif pada data evaluasi [7].

R² sebesar 0,954 berarti model menjelaskan sekitar 95,4% variasi target pada data uji menurut definisi koefisien determinasi. Angka tersebut **bukan berarti akurasi klasifikasi sebesar 95,4%**, dan tidak berarti seluruh prediksi memiliki kesalahan sebesar 4,6%.
R² tidak memiliki satuan. Transformasi linear yang sama pada nilai aktual dan prediksi tidak mengubah nilai R², berbeda dengan MAE dan MSE yang mengikuti skala target. Namun, hal ini tidak menghilangkan potensi kebocoran informasi akibat preprocessing yang dilakukan sebelum pembagian data.

**d. Cara membaca ketiga metrik secara bersamaan**

| Metrik | Kondisi yang diharapkan | Makna utama |
|---|---|---|
| MAE | Semakin kecil semakin baik | Rata-rata selisih absolut prediksi |
| MSE | Semakin kecil semakin baik | Rata-rata kuadrat kesalahan; lebih menekankan kesalahan besar |
| R² | Semakin tinggi semakin baik; nilai terbaik 1 | Proporsi variasi target yang dijelaskan model; dapat negatif pada data uji |

Model yang dipilih dalam proyek ini adalah model dengan MAE dan MSE lebih kecil serta R² lebih tinggi pada data uji yang sama. Penggunaan beberapa metrik membantu mencegah kesimpulan yang hanya bergantung pada satu ukuran performa [2, hlm. 1305; 3, hlm. 1127–1128].

## 5. Penelitian Terkait

Lima artikel jurnal berikut digunakan sebagai referensi ilmiah utama. Kelima artikel tersebut diterbitkan pada 2024–2025, sehingga memenuhi batas tahun minimal 2022–2023. Jurnal *Journal of Applied Informatics and Computing* (JAIC) dan *MALCOM: Indonesian Journal of Machine Learning and Computer Science* tercatat pada SINTA sebagai jurnal terakreditasi **SINTA 3**. Artikel LARS [6] merupakan referensi jurnal internasional tambahan untuk teori algoritma dan tidak dihitung sebagai bagian dari lima artikel SINTA 3.

| No. | Penelitian | Fokus dan kaitan dengan proyek | Bagian hasil atau teori yang dirujuk |
|---|---|---|---|
| [1] | Ghaisani & Baita (2025) | Prediksi harga mobil bekas Toyota menggunakan KNN Regressor; paling dekat dengan objek penelitian proyek ini | Teori faktor harga dan data mobil, hlm. 194–197; evaluasi, hlm. 198–199; hasil, Tabel VI hlm. 200 |
| [2] | Pradasyah & Baita (2025) | Perbandingan SVR dan LSTM untuk prediksi harga saham; menjadi rujukan penggunaan metrik evaluasi regresi | Bagian II.D “Evaluation”, hlm. 1305; hasil pada Tabel XI, hlm. 1309 |
| [3] | Widiyatmoko dkk. (2025) | Prediksi kenaikan harga beras premium dengan regresi linier; menjadi rujukan persamaan regresi serta MAE, MSE, dan R² | Persamaan regresi, hlm. 1127; MAE, hlm. 1127; MSE dan R², hlm. 1128 |
| [4] | Rochadiani (2024) | Prediksi indeks kualitas udara dengan model ensemble; menjadi rujukan preprocessing dan Gradient Boosting | Preprocessing dan standardisasi, hlm. 385–386; Gradient Boosting dan hasil, hlm. 386–389 |
| [5] | Cahyo & Sulistyono (2025) | Perbandingan regresi linier berganda dan Random Forest; menjadi rujukan regresi, pembagian data, dan evaluasi | Regresi linier dan split data, hlm. 3482; evaluasi/cross-validation, hlm. 3483; hasil, hlm. 3487–3488 |

**Catatan perbandingan:** nilai MAE, MSE, dan R² dari penelitian terdahulu tidak dapat dibandingkan langsung dengan nilai proyek ini karena dataset, jumlah data, satuan target, pembagian data, dan konfigurasi model berbeda. Penelitian terkait digunakan sebagai landasan teori serta pembanding pendekatan, bukan sebagai tolok ukur angka performa secara langsung.

## 6. Hasil dan Evaluasi Model

Hasil evaluasi model pada data uji berdasarkan output notebook:

| Algoritma | MAE | MSE | R² | Keterangan |
|---|---:|---:|---:|---|
| Least Angle Regression (LARS) | 0,5708 | 0,5643 | 0,4415 | Performa terendah di antara tiga model yang diuji |
| Linear Regression (LR) | 0,3536 | 0,2370 | 0,7654 | Lebih baik daripada LARS pada ketiga metrik |
| **Gradient Boosting Regressor (GBR)** | **0,1537** | **0,0465** | **0,9540** | **Performa terbaik berdasarkan hasil pengujian saat ini** |

Model GBR disimpan ke dalam berkas:

- `gbr_model.joblib`
- `gbr_model.pkl`

**Penting:** angka MAE dan MSE pada tabel di atas mengikuti target `price` yang telah distandardisasi oleh notebook. Jadi, MAE 0,1537 bukan berarti kesalahan rata-rata sebesar 0,1537 GBP. Untuk melaporkan MAE dalam GBP, evaluasi perlu dilakukan menggunakan nilai target dan prediksi yang telah dikembalikan ke skala GBP, atau target `price` tidak distandardisasi.

## 7. Pembahasan dan Keterbatasan

1. **Model dengan hasil terbaik.** GBR menghasilkan MAE dan MSE paling kecil serta R² tertinggi di antara tiga algoritma yang diuji. Berdasarkan metrik pada data uji yang sama, GBR menjadi model terbaik pada eksperimen ini.
2. **Perbedaan pendekatan model.** LARS dan Linear Regression merupakan pendekatan regresi linier, sedangkan GBR membangun sejumlah pohon keputusan secara bertahap. Hasil yang lebih tinggi pada GBR menunjukkan bahwa pendekatan ensemble berbasis pohon lebih sesuai untuk pola yang ditemukan pada dataset eksperimen ini. Hasil tersebut tidak berarti GBR selalu lebih baik untuk semua dataset [4, hlm. 388–389].
3. **Arti R².** Nilai R² GBR sebesar 0,9540 menunjukkan bahwa model menjelaskan sekitar 95,4% variasi target pada data uji berdasarkan perhitungan R². Angka tersebut bukan persentase akurasi klasifikasi.
4. **Skala MAE dan MSE.** Notebook menjalankan `StandardScaler` pada kolom numerik yang juga mencakup `price`. Oleh sebab itu, MAE dan MSE yang ditampilkan merupakan nilai pada skala target yang telah distandardisasi, bukan skala GBP.
5. **Potensi data leakage.** `StandardScaler.fit_transform()` dijalankan sebelum `train_test_split`, sehingga rata-rata dan standar deviasi dihitung dari keseluruhan data, termasuk bagian yang kemudian menjadi data uji. Selain itu, penyaringan outlier dilakukan sebelum pemisahan data dan diterapkan pada kolom numerik yang mencakup `price`. Kondisi ini dapat membuat evaluasi data uji tidak sepenuhnya independen dari proses preprocessing.
6. **Encoding yang benar-benar dipakai.** Notebook membuat dataframe One-Hot Encoding, tetapi fitur yang akhirnya digunakan untuk pemodelan berasal dari dataframe hasil LabelEncoder.
7. **Pengembangan selanjutnya.** Untuk evaluasi yang lebih kuat, pemisahan data sebaiknya dilakukan terlebih dahulu. Imputasi dan standardisasi fitur kemudian dipelajari hanya dari data latih, misalnya melalui `Pipeline` atau `ColumnTransformer`. Target `price` sebaiknya tetap dalam GBP ketika menghitung MAE dan MSE, atau hasil prediksi target yang distandardisasi harus dikembalikan dengan `inverse_transform` sebelum metrik dalam GBP dilaporkan. Penggunaan *cross-validation* dan pencarian hyperparameter juga dapat dipertimbangkan [5, hlm. 3483].

## 8. Peta Sumber Kutipan

Tabel ini menunjukkan lokasi yang dapat diperiksa dosen pada sumber asli. Nomor halaman merujuk pada nomor halaman yang tercetak di artikel/PDF, bukan urutan halaman saat PDF dibuka. Semua penjelasan teori di README merupakan parafrase, kecuali persamaan matematis yang ditulis ulang dengan notasi yang diseragamkan.

| No. | Materi/pernyataan dalam README | Sumber | Lokasi spesifik pada sumber |
|---|---|---|---|
| 1 | Harga mobil bekas serta faktor yang berkaitan dengan harga (model, tahun, jarak tempuh, dan fitur kendaraan) | [1] | Bagian I “Introduction”, hlm. 194–195 |
| 2 | Kasus prediksi harga mobil bekas Toyota dengan machine learning dan proses eksplorasi data | [1] | Bagian II dan III, hlm. 196–198 |
| 3 | Regresi sebagai tugas prediksi target numerik; perbedaan metrik klasifikasi dan regresi | [2] | Bagian II.D “Evaluation”, hlm. 1305 |
| 4 | Pemeriksaan missing value dan preprocessing | [1]; [4] | [1] Bagian II, hlm. 196; [4] Bagian II.B “Data Preprocessing”, hlm. 385 |
| 5 | Boxplot dan pemeriksaan pencilan | [4] | Bagian II.B “Data Preprocessing”, hlm. 385 |
| 6 | Rumus IQR dan batas pencilan | [1] | Bagian II.B, hlm. 197 |
| 7 | Label Encoding dan One-Hot Encoding untuk variabel kategorikal | [5] | Bagian II.B “Preprocessing”, hlm. 3481–3482 |
| 8 | Standardisasi/Z-score dan alasan penyamaan skala | [4]; [5] | [4] Bagian II.D “Feature Engineering”, hlm. 386; [5] Bagian II.B, hlm. 3482, Persamaan (2) |
| 9 | EDA, histogram, dan correlation matrix | [4]; [5] | [4] Bagian III, hlm. 386; [5] Bagian III.C, hlm. 3484–3485 |
| 10 | Pembagian training set dan testing set 80:20 | [5] | Bagian II.B, hlm. 3482 |
| 11 | Cross-validation untuk evaluasi yang lebih stabil | [5] | Bagian II.I “Evaluation”, hlm. 3483 |
| 12 | Definisi regresi linier berganda dan model matematisnya | [3]; [5] | [3] hlm. 1126–1127, Persamaan (2); [5] Bagian II.G, hlm. 3482 |
| 13 | Konsep dan langkah algoritma LARS | [6] | Bagian “Annex 3: Least-Angle Regression (LARS) Algorithm” pada versi online artikel; artikel terbit di *Journal of Forecasting*, Vol. 41(6), hlm. 1181–1200 |
| 14 | Konsep Gradient Boosting: weak learner, penambahan pohon tiap iterasi | [4] | Bagian II.E “Modelling”, subbagian Gradient Boosting Regressor, hlm. 386 |
| 15 | Pohon baru memperbaiki kesalahan pohon sebelumnya; kemampuan menangkap hubungan nonlinier | [4] | Bagian III.B dan pembahasan, hlm. 388–389 |
| 16 | Definisi MAE dan rumus | [3]; [1] | [3] hlm. 1127, Persamaan (3); [1] Bagian II.E “Evaluation Model”, hlm. 198 |
| 17 | Definisi MSE dan rumus; penalti lebih besar untuk kesalahan besar | [3]; [1] | [3] hlm. 1128, Persamaan (4); [1] Bagian II.E “Evaluation Model”, hlm. 198 |
| 18 | R², rumus residual sum of squares dan total sum of squares | [3] | hlm. 1128, Persamaan (5) |
| 19 | Interpretasi R² yang dapat negatif pada data uji | [7] | Dokumentasi scikit-learn, bagian “R² score, the coefficient of determination” |
| 20 | Kriteria model terbaik: MAE/MSE lebih kecil dan R² lebih tinggi | [3]; [5] | [3] hlm. 1128; [5] Bagian II.I “Evaluation”, hlm. 3483 |
| 21 | Hasil penelitian prediksi harga Toyota dengan KNN | [1] | Abstrak hlm. 194 dan Tabel VI hlm. 200 |
| 22 | Hasil perbandingan SVR dan LSTM | [2] | Abstrak hlm. 1301 dan Tabel XI hlm. 1309 |
| 23 | Hasil regresi linier untuk harga beras premium | [3] | Abstrak hlm. 1125 |
| 24 | Hasil Gradient Boosting pada prediksi indeks kualitas udara | [4] | Abstrak hlm. 384 dan Tabel I hlm. 388 |
| 25 | Hasil perbandingan regresi linier dan Random Forest | [5] | Abstrak hlm. 3479 dan Tabel IV hlm. 3487 |

**Catatan sumber sekunder:** apabila artikel jurnal mengutip teori dari referensi lain, README ini tetap mencantumkan artikel jurnal yang dibaca langsung beserta bagian/halaman tempat pembahasan tersebut ditemukan. Referensi primer LARS [8] dicantumkan sebagai rujukan asal algoritma, sedangkan [6] menjadi rujukan jurnal tahun 2022 yang menjelaskan algoritma LARS pada bagian lampirannya.

## 9. Daftar Pustaka

**Lima artikel jurnal utama (terbit 2024–2025) dari jurnal terakreditasi SINTA 3:**

[1] Ghaisani, M. S., & Baita, A. (2025). Implementation of the K-Nearest Neighbors (KNN) Regressor method to predict Toyota used car prices. *Journal of Applied Informatics and Computing, 9*(1), 194–201. https://doi.org/10.30871/jaic.v9i1.8860. Jurnal: JAIC (SINTA 3).

[2] Pradasyah, A., & Baita, A. (2025). Comparative study of Support Vector Regression and Long Short-Term Memory for stock price prediction. *Journal of Applied Informatics and Computing, 9*(4), 1301–1311. https://doi.org/10.30871/jaic.v9i4.9425. Jurnal: JAIC (SINTA 3).

[3] Widiyatmoko, A. T., Butsianto, S., & Nugroho, A. (2025). Penerapan machine learning untuk prediksi kenaikan harga beras premium menggunakan algoritma regresi linier. *MALCOM: Indonesian Journal of Machine Learning and Computer Science, 5*(3), 1125–1132. https://doi.org/10.57152/malcom.v5i3.2123. Jurnal: MALCOM (SINTA 3).

[4] Rochadiani, T. H. (2024). Prediction of air quality index using ensemble models. *Journal of Applied Informatics and Computing, 8*(2), 384–389. https://jurnal.polibatam.ac.id/index.php/JAIC/article/download/8532/2497. Jurnal: JAIC (SINTA 3).

[5] Cahyo, S. A. N., & Sulistyono, M. Y. T. (2025). Comparison of multiple linear regression and random forest methods for predicting national rice production in Indonesia. *Journal of Applied Informatics and Computing, 9*(6), 3479–3489. https://doi.org/10.30871/jaic.v9i6.11398. Jurnal: JAIC (SINTA 3).

**Referensi jurnal tambahan untuk algoritma LARS (2022; bukan bagian dari hitungan lima artikel SINTA 3):**

[6] Jardet, C., & Meunier, B. (2022). Nowcasting world GDP growth with high-frequency data. *Journal of Forecasting, 41*(6), 1181–1200. https://doi.org/10.1002/for.2858.

**Dokumentasi teknis:**

[7] Scikit-learn developers. *R² score, the coefficient of determination*. Scikit-learn documentation. https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination.

**Referensi primer algoritma LARS:**

[8] Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *The Annals of Statistics, 32*(2), 407–499. https://doi.org/10.1214/009053604000000067.

**Laman pemeriksaan akreditasi jurnal:**

- [SINTA – Journal of Applied Informatics and Computing (JAIC)](https://sinta.kemdiktisaintek.go.id/journals/profile/4696)
- [SINTA – MALCOM: Indonesian Journal of Machine Learning and Computer Science](https://sinta.kemdiktisaintek.go.id/journals/profile/11292)

## 10. Struktur Repositori

```text
├── Praktek.ipynb        # Notebook proses data, EDA, modelling, dan evaluasi
├── toyota.csv           # Dataset mobil bekas Toyota
├── README.md            # Dokumentasi proyek
├── gbr_model.joblib     # Model GBR (dihasilkan setelah notebook dijalankan)
└── gbr_model.pkl        # Model GBR (dihasilkan setelah notebook dijalankan)
```

Berkas model hanya tersedia setelah notebook dijalankan sampai tahap penyimpanan model.

## 11. Cara Menjalankan Proyek

### 1. Clone repositori

```bash
git clone https://github.com/Willkycmz/toyota-used-car-price-prediction-with-Regession.git
cd toyota-used-car-price-prediction-with-Regession
```

### 2. Buat dan aktifkan virtual environment (opsional)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependensi

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib jupyter
```

### 4. Jalankan notebook

```bash
jupyter notebook Praktek.ipynb
```

## 12. Kontributor

- William Cemas ([@Willkycmz](https://github.com/Willkycmz))
