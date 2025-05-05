import cv2
import os
import sqlite3
from tkinter import messagebox

# ----------- Config ----------
DB_PATH = "students.db"
HAAR_CASCADE_FACE = "haarcascades/haarcascade_frontalface_default.xml"
HAAR_CASCADE_EYE = "haarcascades/haarcascade_eye.xml"
# -----------------------------

def get_student_data_by_regno(reg_no):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE reg_no = ?", (reg_no,))
    student_data = cursor.fetchone()
    conn.close()
    return student_data

def capture_face_data(face_data_path):
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FACE)
    eye_cascade = cv2.CascadeClassifier(HAAR_CASCADE_EYE)
    cap = cv2.VideoCapture(0)
    face_count = 0
    blink_detected = False

    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not detected!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_region_gray = gray[y:y+h, x:x+w]
            face_region_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(face_region_gray, scaleFactor=1.1, minNeighbors=5)

            if len(eyes) >= 2:
                blink_detected = True
            else:
                if blink_detected:
                    # Only save after blink detected once
                    face_count += 1
                    face_image = cv2.resize(face_region_color, (200, 200))
                    face_image_path = os.path.join(face_data_path, f"face_{face_count}.png")
                    cv2.imwrite(face_image_path, face_image)
                    print(f"Captured face {face_count}")
                    blink_detected = False  # Reset blink detection

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, f"Captured: {face_count}/20", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Capturing Faces (Blink your eyes)', frame)

        if face_count >= 20:
            messagebox.showinfo("Success", f"Captured {face_count} face images successfully!")
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_face_capture(reg_no):
    student_data = get_student_data_by_regno(reg_no)
    if student_data:
        face_data_path = student_data[9]  # Correct: index 8 = face_data_dir
        if not face_data_path:
            messagebox.showerror("Error", "Face data path is missing!")
            return

        if not os.path.exists(face_data_path):
            os.makedirs(face_data_path)

        capture_face_data(face_data_path)
    else:
        messagebox.showerror("Error", "Student not found!")

if __name__ == "__main__":
    reg_no = input("Enter Registration Number for Face Capture: ")
    start_face_capture(reg_no)
