import sqlite3

# Connect to the database
conn = sqlite3.connect('data/carbon_data.db')
cursor = conn.cursor()

# Create the comments table
cursor.execute('''
CREATE TABLE comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    text TEXT NOT NULL,
    rating INTEGER NOT NULL
)
''')

# Commit the transaction and close the connection
conn.commit()
conn.close()

print("Comments table created successfully.")