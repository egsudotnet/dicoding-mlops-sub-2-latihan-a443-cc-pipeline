# Submission 2: Stress Detection

Nama: Egih Sugiatna

Username dicoding: egihsugiatna

![1737442405702](https://storage.googleapis.com/kaggle-datasets-images/2961947/5100130/dddc8ae8e2864dcc95d830467e023383/dataset-cover.jpg?t=2023-03-03-14-07-22)

|                         | Deskripsi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                 | [Human Stress Prediction](https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Masalah                 | Dataset yang digunakan adalah data teks dengan label biner, yang menunjukkan apakah teks tersebut mencerminkan kondisi stres (1) atau tidak stres (0). Dataset berisi teks deskriptif tentang pengalaman atau situasi tertentu.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Solusi machine learning | Solusi yang diusulkan adalah membangun model pembelajaran mesin berbasis Natural Language Processing (NLP) untuk mengklasifikasikan teks ke dalam dua kategori: "stres" atau "tidak stres". Model ini dapat digunakan dalam sistem pendukung kesehatan mental untuk membantu identifikasi dini masalah stres dari teks yang diberikan.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Metode pengolahan       | Kode ini dimulai dengan mendownload data dari Kaggle, mengambil dua kolom penting, yaitu*text* dan  *label* , yang kemudian disimpan dalam file CSV baru. Selanjutnya, data teks diproses menggunakan TensorFlow Transform (tft) untuk menyiapkan data untuk pelatihan model. Kolom *text* diubah menjadi representasi numerik menggunakan lapisan `TextVectorization`, yang melakukan normalisasi teks seperti mengubah huruf besar menjadi kecil dan menghapus tanda baca, serta menghasilkan representasi integer berdasarkan tokenisasi. Data numerik ini digunakan dalam proses embedding, menghasilkan representasi vektor berdimensi rendah untuk input model pembelajaran mesin. Pipeline ini mengintegrasikan preproses data, pelatihan, dan evaluasi model untuk menyelesaikan tugas klasifikasi teks dengan efisiensi tinggi.                                                                                                                                                                                                                                       |
| Arsitektur model        | Arsitektur model yang digunakan dalam skrip ini adalah model deep learning berbasis TensorFlow dengan pendekatan embedding untuk pengolahan teks. Model dimulai dengan input lapisan teks, di mana data teks mentah diolah melalui lapisan `TextVectorization` untuk tokenisasi dan vektorisasi teks. Representasi numerik yang dihasilkan kemudian diubah menjadi vektor berdimensi rendah menggunakan lapisan embedding. Selanjutnya, lapisan `GlobalAveragePooling1D` digunakan untuk mereduksi dimensi vektor embedding, menghasilkan representasi global yang lebih ringkas. Representasi ini diteruskan melalui dua lapisan `Dense` berturut-turut, masing-masing dengan aktivasi ReLU, untuk mengekstraksi fitur non-linear yang relevan. Akhirnya, lapisan output `Dense` dengan aktivasi sigmoid digunakan untuk menghasilkan prediksi probabilitas biner, sesuai dengan tugas klasifikasi dua kelas. Model ini dioptimalkan menggunakan fungsi loss *binary crossentropy* dan algoritma  *Adam optimizer* , dengan metrik akurasi biner untuk evaluasi performa. |
| Metrik evaluasi         | Metrik yang digunakan untuk mengevaluasi performa model adalah*binary accuracy* , yang mengukur persentase prediksi yang benar dalam tugas klasifikasi biner. Metrik ini membandingkan label asli dengan prediksi model untuk menentukan apakah keduanya sesuai. Jika probabilitas keluaran dari lapisan output sigmoid lebih besar dari 0,5, model akan mengklasifikasikan hasil sebagai kelas positif, dan sebaliknya untuk kelas negatif. Dengan menghitung persentase prediksi yang sesuai dengan label asli, metrik ini memberikan gambaran langsung tentang sejauh mana model mampu membedakan antara dua kelas dengan benar. *Binary accuracy* dipilih karena sederhana dan relevan untuk tugas klasifikasi biner yang merupakan fokus utama dari model ini.                                                                                                                                                                                                                                                                                                                |
| Performa model          | Model yang dibangun menunjukkan performa awal yang baik dengan*binary accuracy* pada data pelatihan mencapai 89,82%, menunjukkan kemampuan model untuk membedakan dua kelas dengan tingkat keakuratan yang tinggi. Namun, pada data validasi, *binary accuracy* menurun menjadi 73,57%, yang diikuti dengan *val_loss* sebesar 1,3422. Penurunan akurasi dan tingginya *validation loss* menunjukkan adanya potensi  *overfitting* , di mana model mampu belajar dengan baik pada data pelatihan tetapi kurang mampu menggeneralisasi pada data validasi. Hal ini mengindikasikan perlunya evaluasi lebih lanjut terhadap arsitektur model, hiperparameter, atau representasi data untuk meningkatkan kemampuan generalisasi model pada data baru.                                                                                                                                                                                                                                                                                                                           |
| Opsi deployment         | Deployment model yang diusulkan adalah menggunakan model serving mnggunakan layanan cloud seperti heroku dan railway.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Web app                 | [stress-detection](https://es-stress-58b7d6b8fb0e.herokuapp.com/v1/models/stress-model/metadata)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Monitoring              | Monitoring sistem Machine Learning (ML) dengan**Prometheus** berguna untuk melacak performa model, penggunaan sumber daya (CPU, RAM, GPU), latensi inferensi, serta kesehatan layanan secara real-time. Dengan Prometheus, kita bisa mengumpulkan metrik dari model ML, mendeteksi anomali, dan mengatasi bottleneck sebelum berdampak besar. Keuntungan utamanya adalah **pemantauan otomatis**, **alerting yang proaktif**, serta **integrasi mudah dengan tools seperti Grafana** untuk visualisasi, sehingga meningkatkan efisiensi operasional dan keandalan sistem ML.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
#   m l o p s - h e r o k u  
 