import os
os.makedirs("data", exist_ok=True)

import sqlite3

# Veritabanı yolunu kendi path'ine göre değiştir
db_path = r"D://my_agent_win//data//persona.sqlite"

try:
    conn = sqlite3.connect(db_path)
    print("Veritabanı açıldı başarılı!")
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    print("Tablolar:", tables)
    conn.close()
except Exception as e:
    print("Hata:", e)
