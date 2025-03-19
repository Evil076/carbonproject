import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Load the processed data from the database
    cursor.execute('SELECT * FROM processed_carbon_data')
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)
    X = df.drop('carbon_level', axis=1)
    y = df['carbon_level']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'models/carbon_model.pkl')

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()