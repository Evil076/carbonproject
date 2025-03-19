import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def create_database():
    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS carbon_data (
        id INTEGER PRIMARY KEY,
        feature1 REAL,
        feature2 REAL,
        carbon_level REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS new_carbon_data (
        id INTEGER PRIMARY KEY,
        feature1 REAL,
        feature2 REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predicted_carbon_data (
        id INTEGER PRIMARY KEY,
        feature1 REAL,
        feature2 REAL,
        predicted_carbon_level REAL
    )
    ''')

    # Insert sample data
    cursor.execute('''
    INSERT INTO carbon_data (feature1, feature2, carbon_level)
    VALUES
    (1.0, 2.0, 10.0),
    (2.0, 3.0, 20.0),
    (3.0, 4.0, 30.0),
    (4.0, 5.0, 40.0),
    (5.0, 6.0, 50.0),
    (6.0, 7.0, 60.0),
    (7.0, 8.0, 70.0),
    (8.0, 9.0, 80.0),
    (9.0, 10.0, 90.0),
    (10.0, 11.0, 100.0)
    ''')

    cursor.execute('''
    INSERT INTO new_carbon_data (feature1, feature2)
    VALUES
    (11.0, 12.0),
    (12.0, 13.0)
    ''')

    # Commit and close the connection
    conn.commit()
    conn.close()

def train_model():
    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Load historical data
    cursor.execute('SELECT * FROM carbon_data')
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Exclude the 'id' column from the features
    X = df[['feature1', 'feature2']]
    y = df['carbon_level']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model to a file
    joblib.dump(model, 'models/carbon_model.pkl')

    # Close the connection
    conn.close()

if __name__ == "__main__":
    create_database()
    train_model()