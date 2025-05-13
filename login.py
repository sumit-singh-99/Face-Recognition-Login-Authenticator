import sqlite3
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import os
import face_recognition
import time

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
    match_threshold = 0.38  # Stricter threshold

    face_data_path = os.path.join("face_data", reg_no)

    if not os.path.exists(face_data_path):
        messagebox.showerror("Error", f"No face data found for Registration No: {reg_no}")
        cap.release()
        return False

    known_encodings = []
    for filename in os.listdir(face_data_path):
        img_path = os.path.join(face_data_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img_rgb, num_jitters=2)
            if encodings:
                known_encodings.append(encodings[0])

    if not known_encodings:
        messagebox.showerror("Error", "No valid face images found!")
        cap.release()
        return False

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_distance = np.min(face_distances)

            confidence = 1 - best_distance

            if best_distance < match_threshold:
                recognized = True
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Matched ({confidence:.2f})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                break
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, f"Unmatched ({confidence:.2f})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if recognized or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Stop after 20 seconds if no match
        if time.time() - start_time > 20:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not recognized:
        messagebox.showwarning("Face Not Recognized", "Face didn't match. Please try again!")

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

# ------------------ GUI ---------------------
root = tk.Tk()
root.title("Student Login")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Student Login", font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=20)

tk.Label(root, text="Registration No:", bg="#f0f0f0", font=("Helvetica", 12)).pack()
entry_regno = tk.Entry(root, font=("Helvetica", 12))
entry_regno.pack(pady=5)

tk.Label(root, text="Password:", bg="#f0f0f0", font=("Helvetica", 12)).pack()
entry_password = tk.Entry(root, font=("Helvetica", 12), show="*")
entry_password.pack(pady=5)

tk.Button(root, text="Login", font=("Helvetica", 12), bg="#4CAF50", fg="white", width=15, command=login).pack(pady=20)

root.mainloop()
