# Analisis Sentimen pada Tweet Ulasan Film - NLP

- Author: Rijal Ghodi
- Email: rijalgdev@gmail.com
- Github: https://github.com/rijalghodi

- Github Project: https://github.com/rijalghodi/movie-review-sentiment-analysis
- Kaggle Project: https://www.kaggle.com/code/zalcode/analisis-sentimen-pada-tweet-ulasan-film

<img src="https://img.freepik.com/free-photo/view-3d-cinema-elements_23-2150720822.jpg" width=400>

## Tujuan

Tujuan proyek ini adalah untuk membangun model dengan akurasi terbaik untuk memprediksi sentimen yang terkandung pada tweet ulasan film

## Dataset

Dataset dapat di-download [di sini](https://raw.githubusercontent.com/rijalghodi/datasets/main/movie_review.csv)

## Fitur-Fitur Dataset

**Dataset Features:**

- `id`: identifier unik setiap tweet
- `Sentiment`: Label sentimen (negative, positive, neutral) atas film
- `Text Tweet`: Teks tweet yang mengulas film

# Stategi Penyelesaian

1. **Persiapan Awal**: Memulai dengan mengimpor library yang diperlukan dan memuat dataset ke dalam lingkungan kerja.

2. **Analisis Data Eksploratif (EDA)**: Untuk memahami statistik dasar dari dataset, seperti distribusi variabel.

3. **Feature Engineering**: Tahap penting ini melibatkan pre-processing teks dan mengkodekan label.

4. **Pembangunan Model Dasar (Baseline)**: Membuat model awal yang sederhana.

5. **Optimisasi Hyperparameter**: Proses ini melibatkan penyetelan parameter-model (hyperparameters) untuk mencapai kinerja optimal. Metode seperti pencarian grid atau pencarian acak digunakan untuk menemukan kombinasi parameter terbaik.

6. **Implementasi dan Demo**: Terakhir, model yang telah dibangun akan diimplementasikan dan diuji menggunakan data ulasan film nyata untuk memvalidasi kinerjanya dalam konteks yang relevan.

# Strategi Preprocessing

Text pre-processing adalah serangkaian teknik untuk membersihkan, mengubah format, dan mengubah teks mentah menjadi representasi yang lebih terstruktur dan dapat diolah dengan mudah dan efisien.

Langkah-langkah yang digunakan dalam melakukan text preprocessing adalah sebagai berikut:

- Cleaning text
- Remove stopwords
- Stemming / Lemmatization

## Cleaning Text

Tahap ini menghilangkan teks-teks yang tidak relevan, seperti link, tagar, tanda baca, emoji, dan mengubah semua huruf menjadi huruf kecil

## Menghilangkan Stopword

Stopword adalah kata-kata yang umumnya diabaikan dalam proses analisis teks karena kurangnya informasi atau makna yang signifikan. Contohnya, dalam bahasa Indonesia, kata-kata seperti "dan", "atau", "yang", dan "di". Untuk menghasilkan model yang efisien dan akurat, kita harus menghilangkan stopword.

## Stemming / Lemmatization

Stemming: Ini adalah proses normalisasi kata dengan menghapus imbuhan (affixes) dari kata untuk menghasilkan bentuk dasar, yang disebut lemma. Contohnya, kata "berlari" dan "menemukan" akan disederhanakan menjadi "lari", "temu".

Lemmatization juga merupakan proses normalisasi kata-kata, tetapi lebih kompleks daripada stemming karena melibatkan analisis morfologi kata-kata untuk mengembalikan kata-kata ke bentuk dasarnya, yang disebut "lemma". Contohnya, kata "mempertanggung-jawabkan" akan disederahanakan menjadi "tanggung-jawab"

Keduanya seringkali menghasilkan kata dasar yang sama tapi dalam beberapa kasus menghasilkan perbedaan.

# Strategi Pembangunan Model Dasar

## Strategi Pembagian Dataset Train dan Test

Dataset dibagi ke dalam train dan test dengan proporsi 8:2.

## Strategi Deep Learning

Masalah yang dihadapi adalah klasifikasi teks. Oleh karena itu, kita akan menggunakan deep learning untuk menyelesaikannya.

Karena teks adalah data sekuen yang sangat sensitif terhadap urutan, kita akan menggunakan layer LSTM dalam deep learning untuk membangun model dengan akurasi yang tinggi.

Untuk model dasar ini, kita akan menggunakan konfigurasi sebagai berikut:

- Layer embedding dengan input_dim = 1000 dan output_dim = 100.
- Layer LSTM dengan units = 64, dropouts = 0.2, dan recurrent_dropout = 0.2.
- Layer output dengan 1 neuron dan fungsi aktivasi softmax.

## Startegi Peningkatan Performa Model

Model dasar menghasilkan akurasi yang sangat rendah (train: 50% dan test: 47%). Untuk meningkatkan akurasi kita perlu melakukan tuning.

Randahnya akurasi ini diduga karena terlalu sedikit node pada layer embedding dan LSTM. Oleh karena itu, kita akan secara bertahap menambah jumlah node pada kedua layer ini.

Untuk itu, kita akan mendefinisikan dua parameter, yaitu 'embedding_output_dim' dengan nilai di antara [32, 64, 128], dan 'lstm_units' dengan nilai [32, 64, 128].

Kami akan menggunakan GridSearchCV untuk mencari parameter terbaik.

## Evaluasi Model

Baik dataset train dan test memiliki akurasi yang sangat rendah (50% dan 47%).
Untuk meningkatkan akurasi kita perlu melakukan tuning.

Setelah dilakukan hyperparameter tuning dengan parameter yang telah didefinsiikan, ditemukan bahwa parameter terbaik adalah sama seperti pada model dasar, yakni: embedding_output_dim : 64 dan lstm_units : 64.

Dari hasil ini, kita bisa mengambil kesimpulan bahwa menaikkan atau mengurangi jumlah node tidak berdampak pada peningkatan akurasi model

Strategi berikutnya yang bisa dilakukan untuk meningkatkan akurasi antara lain:

1. **Penambahan Layer atau Unit**: Tambahkan layer atau unit pada model Anda untuk meningkatkan kompleksitasnya. Lebih banyak layer LSTM atau layer dense tambahan dapat membantu model mempelajari pola yang lebih kompleks.

2. **Regularisasi**: Gunakan teknik regularisasi seperti dropout atau regularisasi L2 untuk mencegah overfitting pada model Anda.

3. **Optimisasi Hyperparameter Lainnya**: Eksperimen dengan nilai hyperparameter lain seperti tingkat dropout, kecepatan pembelajaran, dan jumlah epoch untuk mencari konfigurasi yang optimal.

4. **Penambahan Fitur**: Selain teks mentah, tambahkan fitur tambahan seperti metadata atau fitur teks tambahan yang diekstraksi menggunakan teknik NLP lainnya.

5. **Penggunaan Word Embeddings yang Lebih Baik**: Gunakan word embeddings yang lebih baik seperti Word2Vec, GloVe, atau FastText, atau gunakan pre-trained embeddings untuk meningkatkan representasi kata dalam model.

6. **Menggunakan Model yang Lebih Kompleks**: Cobalah model yang lebih kompleks seperti Convolutional Neural Networks (CNN) atau model Transformer seperti BERT untuk memperoleh pemahaman yang lebih baik tentang teks.

7. **Penyelidikan Lebih Lanjut tentang Data**: Periksa dataset Anda untuk mengetahui apakah terdapat kesalahan label, ketidakseimbangan data, atau masalah lain yang dapat memengaruhi kinerja model.
