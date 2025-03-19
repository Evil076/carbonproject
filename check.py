import os
import sqlite3

def create_database():
    db_path = 'data/carbon_data.db'
    # Check if the directory exists
    if not os.path.exists(os.path.dirname(db_path)):
        print(f"Directory does not exist: {os.path.dirname(db_path)}")
        return
    try:
        conn = sqlite3.connect(db_path)
        print("Database created and connected successfully.")
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")

create_database()
