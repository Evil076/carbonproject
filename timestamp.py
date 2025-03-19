import sqlite3

# Connect to the database
conn = sqlite3.connect('data/carbon_data.db')
cursor = conn.cursor()

# Add the timestamp column to the comments table
cursor.execute('ALTER TABLE comments ADD COLUMN timestamp TEXT')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Timestamp column added successfully.")