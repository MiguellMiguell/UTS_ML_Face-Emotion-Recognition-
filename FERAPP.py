import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2, torch
import numpy as np
from torchvision import transforms
from model_loader import load_model
import os

# Load model
model, EMOTION_LABELS, DEVICE = load_model("resnet18_fer2013.pth")

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Prediksi
def predict_emotion(img):
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return EMOTION_LABELS[pred.item()], conf.item()

# GUI
root = tk.Tk()
root.title("FER2013 Emotion Detection")
root.geometry("600x600")

lbl_result = tk.Label(root, text="Upload an image or open webcam", font=("Arial", 14))
lbl_result.pack(pady=10)

canvas = tk.Label(root)
canvas.pack()

# Upload Gambar
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path: return
    img = Image.open(file_path).convert("RGB")
    label, conf = predict_emotion(img)
    img_tk = ImageTk.PhotoImage(img.resize((300,300)))
    canvas.config(image=img_tk)
    canvas.image = img_tk
    lbl_result.config(text=f"Emotion: {label} ({conf*100:.1f}%)")

btn_upload = tk.Button(root, text="Upload Image", command=upload_image, width=20)
btn_upload.pack(pady=10)

# Webcam Real-time
cap = None
def open_webcam():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Kamera tidak bisa dibuka")
    else:
        print("Kamera berhasil dibuka")
    update_webcam()

def update_webcam():
    global cap
    ret, frame = cap.read()
    if not ret:
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48,48))
    for (x,y,w,h) in faces:
        face = Image.fromarray(gray[y:y+h, x:x+w])
        label, conf = predict_emotion(face)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf*100:.1f}%", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img_tk = ImageTk.PhotoImage(image=img.resize((400,300)))
    canvas.config(image=img_tk)
    canvas.image = img_tk
    root.after(10, update_webcam)

btn_webcam = tk.Button(root, text="Open Webcam", command=open_webcam, width=20)
btn_webcam.pack(pady=10)

root.mainloop()

