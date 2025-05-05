import sqlite3

# Connect to database (or create if not exists)
conn = sqlite3.connect('students.db')
cursor = conn.cursor()

# Create table for student registration
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reg_no TEXT UNIQUE,
    name TEXT,
    father_name TEXT,
    aadhaar TEXT UNIQUE,
    phone TEXT,
    age INTEGER,
    gender TEXT,
    program TEXT,
    face_data_dir TEXT,
    password TEXT
)
''')

print("✅ Database and Table Created Successfully.")
conn.commit()
conn.close()
