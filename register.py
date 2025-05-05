import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import os
import uuid
import random

# ----------- Config ----------
DB_PATH = "students.db"
BASE_FACE_DATA_DIR = "face_data/"
# -----------------------------

def generate_registration_number():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    while True:
        reg_no = "REG" + str(random.randint(1000000000, 9999999999))

        # Check if reg_no already exists
        cursor.execute("SELECT COUNT(*) FROM students WHERE reg_no = ?", (reg_no,))
        (count,) = cursor.fetchone()

        if count == 0:
            conn.close()
            return reg_no  # Unique reg_no found, return it
        # else loop again to generate new random reg_no


def generate_password():
    return str(uuid.uuid4())[:8]

def save_to_db(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO students (reg_no, name, father_name, aadhaar, phone, age, gender, program,
                                   face_data_dir, password)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        messagebox.showerror("Database Error", f"Aadhaar or RegNo already exists.\n{e}")
        return False
    finally:
        conn.close()

def on_submit():
    name = name_var.get()
    father = father_var.get()
    aadhaar = aadhaar_var.get()
    phone = phone_var.get()
    age = age_var.get()
    gender = gender_var.get()
    program = program_var.get()

    if not all([name, father, aadhaar, phone, age, gender, program]):
        messagebox.showerror("Missing Info", "Please fill out all required fields.")
        return

    reg_no = generate_registration_number()
    password = generate_password()

    # Create student's face data folder
    face_data_path = os.path.join(BASE_FACE_DATA_DIR, reg_no)
    os.makedirs(face_data_path, exist_ok=True)

    # Save to database
    student_data = (
        reg_no, name, father, aadhaar, phone, age, gender, program,
        face_data_path, password
    )

    if save_to_db(student_data):
        messagebox.showinfo("Success", f"Student Registered!\nRegNo: {reg_no}\nPassword: {password}")
        root.destroy()
        import face_capture
        face_capture.start_face_capture(reg_no)
    else:
        return

# --- GUI Setup ---
root = tk.Tk()
root.title("Student Registration")
root.geometry("600x550")

tk.Label(root, text="Student Registration Form", font=("Arial", 18)).pack(pady=10)

form_frame = tk.Frame(root)
form_frame.pack(pady=10)

name_var = tk.StringVar()
father_var = tk.StringVar()
aadhaar_var = tk.StringVar()
phone_var = tk.StringVar()
age_var = tk.StringVar()
gender_var = tk.StringVar()
program_var = tk.StringVar()

fields = [
    ("Full Name", name_var),
    ("Father's Name", father_var),
    ("Aadhaar Number", aadhaar_var),
    ("Phone Number", phone_var),
    ("Age", age_var),
]

for i, (label, var) in enumerate(fields):
    tk.Label(form_frame, text=label).grid(row=i, column=0, sticky="w", padx=10, pady=5)
    tk.Entry(form_frame, textvariable=var, width=40).grid(row=i, column=1, padx=10)

# Gender
tk.Label(form_frame, text="Gender").grid(row=5, column=0, sticky="w", padx=10, pady=5)
gender_combo = ttk.Combobox(form_frame, textvariable=gender_var, values=["Male", "Female", "Other"], state="readonly")
gender_combo.grid(row=5, column=1, padx=10)

# Program
tk.Label(form_frame, text="Program Enrolled").grid(row=6, column=0, sticky="w", padx=10, pady=5)
program_combo = ttk.Combobox(form_frame, textvariable=program_var,
                             values=["B.Tech", "BBA", "BCA", "MCA", "M.Tech"], state="readonly")
program_combo.grid(row=6, column=1, padx=10)

# Submit Button
tk.Button(root, text="Submit & Start Face Capture", command=on_submit, bg="green", fg="white", width=25).pack(pady=20)

root.mainloop()
