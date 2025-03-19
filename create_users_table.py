import sqlite3

# Connect to the database
conn = sqlite3.connect('data/carbon_data.db')
cursor = conn.cursor()

# Create the users table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

# Commit the transaction and close the connection
conn.commit()
conn.close()

print("Users table created successfully.")