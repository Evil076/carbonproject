import sqlite3
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io

def main():
    # Load the trained model
    model = joblib.load('models/carbon_model.pkl')

    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Load new data for prediction
    cursor.execute('SELECT * FROM new_carbon_data')
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    new_data = pd.DataFrame(data, columns=columns)

    # Exclude the 'id' column from the features
    X_new = new_data[['feature1', 'feature2']]

    # Make predictions
    predictions = model.predict(X_new)

    # Save the predictions to the database
    new_data['predicted_carbon_level'] = predictions
    new_data.to_sql('predicted_carbon_data', conn, if_exists='replace', index=False)

    # Load predicted data for analysis
    cursor.execute('SELECT * FROM predicted_carbon_data')
    predicted_data = cursor.fetchall()
    predicted_columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df_predicted = pd.DataFrame(predicted_data, columns=predicted_columns)

    # Perform analysis on the predicted data
    X_predicted = df_predicted[['feature1', 'feature2']]
    y_predicted = df_predicted['predicted_carbon_level']
    model_predicted = LinearRegression()
    model_predicted.fit(X_predicted, y_predicted)
    trend_predicted = model_predicted.coef_

    # Generate a brief report
    report_predicted = f"""
    Predicted Carbon Level Analysis Report:
    - Number of records: {len(df_predicted)}
    - Average predicted carbon level: {df_predicted['predicted_carbon_level'].mean()}
    - Trend coefficients: {trend_predicted}
    """

    # Plot the predicted data
    plt.figure(figsize=(10, 6))
    plt.scatter(df_predicted['feature1'], df_predicted['predicted_carbon_level'], color='blue', label='Feature 1')
    plt.scatter(df_predicted['feature2'], df_predicted['predicted_carbon_level'], color='green', label='Feature 2')
    plt.plot(X_predicted, model_predicted.predict(X_predicted), color='red', linewidth=2, label='Trend Line')
    plt.xlabel('Features')
    plt.ylabel('Predicted Carbon Level')
    plt.title('Predicted Carbon Level Analysis')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('predicted_carbon_level_analysis.png')

    # Close the connection
    conn.close()

    # Print the report
    print(report_predicted)

if __name__ == "__main__":
    main()