import sqlite3
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os

DB_PATH = "students.db"
HAAR_CASCADE_FACE = "haarcascades/haarcascade_frontalface_default.xml"

def verify_credentials(reg_no, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE reg_no = ? AND password = ?", (reg_no, password))
    user = cursor.fetchone()
    conn.close()
    return user

def recognize_face(reg_no):
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FACE)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not detected!")
        return False

    recognized = False
    match_threshold = 70  # Lower = stricter match

    # Build face data path using reg_no
    face_data_path = os.path.join("face_data", reg_no)

    if not os.path.exists(face_data_path):
        messagebox.showerror("Error", f"No face data found for Registration No: {reg_no}")
        cap.release()
        return False

    known_faces = []
    for filename in os.listdir(face_data_path):
        img_path = os.path.join(face_data_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (200, 200))
            known_faces.append(img)

    if not known_faces:
        messagebox.showerror("Error", "No valid face images found!")
        cap.release()
        return False

    attempts = 0
    max_attempts = 100  # Adjust if needed

    while attempts < max_attempts:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            captured_face = gray[y:y+h, x:x+w]
            captured_face = cv2.resize(captured_face, (200, 200))

            min_diff = float('inf')
            for known_face in known_faces:
                diff = np.sum(cv2.absdiff(known_face, captured_face)) / (200 * 200)
                if diff < min_diff:
                    min_diff = diff

            if min_diff < match_threshold:
                recognized = True
                break

        if recognized:
            break

        attempts += 1

    cap.release()
    cv2.destroyAllWindows()
    return recognized

def login():
    reg_no = entry_regno.get().strip()
    password = entry_password.get().strip()

    if not reg_no or not password:
        messagebox.showerror("Error", "Please enter both Registration No. and Password.")
        return

    user = verify_credentials(reg_no, password)
    if not user:
        messagebox.showerror("Error", "Invalid Registration No. or Password.")
        return

    messagebox.showinfo("Info", "Credentials verified! Now verifying face...")

    if recognize_face(reg_no):
        messagebox.showinfo("Success", "Login Successful! 🎯")
    else:
        messagebox.showerror("Error", "Face Authentication Failed! Try again.")

# --------------- Tkinter GUI -------------------
root = tk.Tk()
root.title("Student Login")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

# Title
tk.Label(root, text="Student Login", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)

# Registration Number
tk.Label(root, text="Registration No:", bg="#f0f0f0", font=("Helvetica", 12)).pack()
entry_regno = tk.Entry(root, font=("Helvetica", 12))
entry_regno.pack(pady=5)

# Password
tk.Label(root, text="Password:", bg="#f0f0f0", font=("Helvetica", 12)).pack()
entry_password = tk.Entry(root, font=("Helvetica", 12), show="*")
entry_password.pack(pady=5)

# Login Button
tk.Button(root, text="Login", font=("Helvetica", 12), bg="#4CAF50", fg="white", width=15, command=login).pack(pady=20)

root.mainloop()
