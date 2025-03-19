import os
import sqlite3

def create_database():
    db_path = 'data/carbon_data.db'
    directory = os.path.dirname(db_path)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
        
    try:
        conn = sqlite3.connect(db_path)
        print("Database created and connected successfully.")
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")

create_database()
