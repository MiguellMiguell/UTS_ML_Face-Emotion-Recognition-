# 🧠 Aplikasi Pengenalan Emosi Wajah (FERAPP)

Aplikasi Desktop yang mendeteksi dan mengklasifikasikan emosi wajah manusia secara real-time menggunakan **ResNet18** yang dilatih pada **dataset FER2013**.

Dibangun dengan **Python, PyTorch, Tkinter**, dan **OpenCV**.

---

## 🚀 Fitur
- Deteksi wajah real-time menggunakan webcam.
- Klasifikasi emosi (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
- GUI yang sederhana dan ramah pengguna.
- Aplikasi desktop offline yang dapat dibuat melalui PyInstaller.

---

## 🧩 Teknologi yang digunakan
- **Python 3.10+**
- **PyTorch**
- **OpenCV**
- **Tkinter**
- **Numpy / Pillow**

---

## ⚙️ Instalasi & Pengaturan

1️⃣ Klon Repositori
```bash
git clone https://github.com/MiguellMiguell/UTS_ML_Face-Emotion-Recognition-.git
cd FERAPP

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Download Model File

Letakkan file resnet18_fer2013.pth di folder app/.

4️⃣ Run App
python FERAPP.py

💻 Build to .exe

Jika ingin membuat aplikasi desktop:

pyinstaller --onefile --noconsole ^
  --add-data "haarcascade_frontalface_default.xml;." ^
  --add-data "resnet18_fer2013.pth;." ^
  FERAPP.py


File .exe hasil build akan berada di folder:

dist/FERAPP.exe

🧠 Dataset

Model dilatih menggunakan dataset FER2013, yang berisi 30.000+- gambar wajah berlabel 7 emosi dasar:
Angry 😠
Disgust 🤢
Fear 😨
Happy 😄
Sad 😢
Surprise 😲
Neutral 😐

Dataset diambil dari Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

👥 Team Members
No	Nama Anggota
1.	Miguell 
2.	Franco Poltack Sinaga
3.	Verianto Wijaya
