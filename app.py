from flask import Flask, request, render_template, session, redirect, url_for, send_file, jsonify
import sqlite3
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV  # Add this import
import matplotlib.pyplot as plt
import io
import hashlib
import random
from send_email import send_otp
import time
from fpdf import FPDF
import os
import logging

# Configure logging
logging.basicConfig(filename='otp_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def optimize_hyperparameters(X_train, y_train):
    # Define the model
    model = LinearRegression()

    # Define the hyperparameters to tune
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'positive': [True, False]
    }

    # Set up the GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')

    # Fit the GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    return best_params

@app.route('/')
def home():
    if 'username' not in session:
        return render_template('access_denied.html')
    
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()
    
    # Calculate the average rating
    cursor.execute('SELECT AVG(rating) FROM comments')
    avg_rating = cursor.fetchone()[0]
    
    # Fetch 5 random comments
    cursor.execute('SELECT username, text, rating, timestamp FROM comments ORDER BY RANDOM() LIMIT 5')
    comments = cursor.fetchall()
    
    conn.close()
    
    description = """
    Welcome to the AI Carbon Driven System!
    This project aims to predict and analyze carbon levels based on various features.
    Users can sign up, log in, and provide input data to get predictions on carbon levels.
    Additionally, users can leave comments and ratings about their experience.
    """
    
    return render_template('index.html', comments=comments, description=description, avg_rating=avg_rating)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        conn = sqlite3.connect('data/carbon_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', login_error='Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        email = request.form['email']
        conn = sqlite3.connect('data/carbon_data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        if user:
            conn.close()
            return render_template('signup.html', signup_error='Username already exists')
        cursor.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', (username, password, email))
        conn.commit()
        conn.close()
        otp = random.randint(100000, 999999)
        session['otp'] = otp
        session['username'] = username
        session['email'] = email
        session['otp_timestamp'] = time.time()
        send_otp(email, otp)
        
        # Log the OTP
        logging.info(f'Generated OTP for {email}: {otp}')
        
        return redirect(url_for('verify_otp'))
    return render_template('signup.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        otp = request.form['otp']
        if 'otp' in session and int(otp) == session['otp']:
            session.pop('otp', None)
            session.pop('email', None)
            session.pop('otp_timestamp', None)
            return redirect(url_for('home'))
        else:
            return render_template('verify_otp.html', otp_error='Invalid OTP', email=session.get('email'))
    return render_template('verify_otp.html', email=session.get('email'))

@app.route('/resend_otp', methods=['POST'])
def resend_otp():
    if 'email' in session:
        otp = random.randint(100000, 999999)
        session['otp'] = otp
        session['otp_timestamp'] = time.time()
        send_otp(session['email'], otp)
        
        # Log the OTP
        logging.info(f'Resent OTP for {session["email"]}: {otp}')
        
        return jsonify({'message': 'OTP resent successfully'})
    return jsonify({'message': 'Failed to resend OTP'}), 400

@app.route('/logout')
def logout():
    session.pop('username', None)
    return render_template('logout.html')  # Render the logout confirmation page

    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Get features from the form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])

    # Validate the input values
    if feature1 > 20 or feature2 > 20:
        return render_template('index.html', prediction_text='Error: Feature values must not exceed 20. Please input values less than 20.')

    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Insert the new data into the database
    cursor.execute('''
    INSERT INTO carbon_data (feature1, feature2, carbon_level)
    VALUES (?, ?, ?)
    ''', (feature1, feature2, None))

    # Commit the transaction
    conn.commit()

    # Load the updated data
    cursor.execute('SELECT * FROM carbon_data')
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Exclude rows with NaN values in 'carbon_level'
    df = df.dropna(subset=['carbon_level'])

    # Exclude the 'id' column from the features
    X = df[['feature1', 'feature2']]
    y = df['carbon_level']

    # Retrain the model with the updated data
    model = LinearRegression()
    model.fit(X, y)

    # Save the retrained model
    joblib.dump(model, 'models/carbon_model.pkl')

    # Make predictions
    new_data = pd.DataFrame([[feature1, feature2]], columns=['feature1', 'feature2'])
    prediction = model.predict(new_data)[0]

    # Update the carbon_level in the database with the prediction
    cursor.execute('''
    UPDATE carbon_data
    SET carbon_level = ?
    WHERE feature1 = ? AND feature2 = ?
    ''', (prediction, feature1, feature2))

    # Commit the transaction

    # Calculate the average rating
    cursor.execute('SELECT AVG(rating) FROM comments')
    avg_rating = cursor.fetchone()[0]

    # Fetch 5 random comments
    cursor.execute('SELECT username, text, rating, timestamp FROM comments ORDER BY RANDOM() LIMIT 5')
    comments = cursor.fetchall()

    conn.commit()
    conn.close()

    return render_template('index.html', prediction_text=f'Predicted Carbon Level: {prediction}', avg_rating=avg_rating, comments=comments)

@app.route('/analyze', methods=['GET'])
def analyze():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Load historical data
    cursor.execute('SELECT * FROM carbon_data')
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Exclude rows with NaN values in 'carbon_level'
    df = df.dropna(subset=['carbon_level'])

    # Exclude the 'id' column from the features
    X = df[['feature1', 'feature2']]
    y = df['carbon_level']
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_

    # Determine if carbon level is increasing or decreasing
    trend_text = "Carbon level is increasing" if trend[0] > 0 else "Carbon level is decreasing"

    # Generate a brief report
    report = f"""
    Carbon Level Analysis Report:
    - Number of records: {len(df)}
    - Average carbon level: {df['carbon_level'].mean()}
    - Trend coefficients: {trend}
    - {trend_text}
    """

    # Calculate the average rating
    cursor.execute('SELECT AVG(rating) FROM comments')
    avg_rating = cursor.fetchone()[0]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(df['feature1'], df['carbon_level'], color='blue', label='Feature 1')
    plt.scatter(df['feature2'], df['carbon_level'], color='green', label='Feature 2')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Trend Line')
    plt.xlabel('Features')
    plt.ylabel('Carbon Level')
    plt.title('Carbon Level Analysis')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Close the connection
    conn.close()

    return render_template('index.html', report_text=report, plot_url='/plot.png', trend_text=trend_text, avg_rating=avg_rating)

@app.route('/plot.png')
def plot_png():
    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Load historical data
    cursor.execute('SELECT * FROM carbon_data')
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Exclude rows with NaN values in 'carbon_level'
    df = df.dropna(subset=['carbon_level'])

    # Exclude the 'id' column from the features
    X = df[['feature1', 'feature2']]
    y = df['carbon_level']
    model = LinearRegression()
    model.fit(X, y)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(df['feature1'], df['carbon_level'], color='blue', label='Feature 1')
    plt.scatter(df['feature2'], df['carbon_level'], color='green', label='Feature 2')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Trend Line')
    plt.xlabel('Features')
    plt.ylabel('Carbon Level')
    plt.title('Carbon Level Analysis')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Close the connection
    conn.close()

    return send_file(img, mimetype='image/png')

@app.route('/download_report')
def download_report():
    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Load historical data
    cursor.execute('SELECT * FROM carbon_data')
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Exclude rows with NaN values in 'carbon_level'
    df = df.dropna(subset=['carbon_level'])

    # Exclude the 'id' column from the features
    X = df[['feature1', 'feature2']]
    y = df['carbon_level']
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_

    # Determine if carbon level is increasing or decreasing
    trend_text = "Carbon level is increasing" if trend[0] > 0 else "Carbon level is decreasing"

    # Generate a brief report
    report = f"""
    Carbon Level Analysis Report:
    - Number of records: {len(df)}
    - Average carbon level: {df['carbon_level'].mean()}
    - Trend coefficients: {trend}
    - {trend_text}
    """

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(df['feature1'], df['carbon_level'], color='blue', label='Feature 1')
    plt.scatter(df['feature2'], df['carbon_level'], color='green', label='Feature 2')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Trend Line')
    plt.xlabel('Features')
    plt.ylabel('Carbon Level')
    plt.title('Carbon Level Analysis')
    plt.legend()
    plt.grid(True)

    # Ensure the static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    # Save the plot to a file
    plot_path = 'static/carbon_level_analysis_plot.png'
    plt.savefig(plot_path)

    # Create a PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "Carbon Level Simulation Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report)
    pdf.image(plot_path, x=10, y=pdf.get_y(), w=pdf.w - 20)  # Use pdf.w instead of pdf.epw

    # Save the PDF to a BytesIO object
    pdf_io = io.BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1')  # Use 'S' to write to the BytesIO object
    pdf_io.write(pdf_output)
    pdf_io.seek(0)

    # Close the connection
    conn.close()

    return send_file(pdf_io, mimetype='application/pdf', as_attachment=True, download_name='carbon_level_analysis_report.pdf')

@app.route('/comment', methods=['POST'])
def comment():
    if 'username' not in session:
        return redirect(url_for('login'))

    comment_text = request.form['comment']
    rating = request.form['rating']
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO comments (username, text, rating, timestamp)
    VALUES (?, ?, ?, ?)
    ''', (session['username'], comment_text, rating, timestamp))
    conn.commit()
    conn.close()

    return redirect(url_for('home'))

@app.route('/evaluate', methods=['GET'])
def evaluate():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Connect to the database
    conn = sqlite3.connect('data/carbon_data.db')
    cursor = conn.cursor()

    # Load historical data
    cursor.execute('SELECT * FROM carbon_data')
    data = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Exclude rows with NaN values in 'carbon_level'
    df = df.dropna(subset=['carbon_level'])

    # Exclude the 'id' column from the features
    X = df[['feature1', 'feature2']]
    y = df['carbon_level']

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = optimize_hyperparameters(X_train, y_train)

    # Train the model
    model = LinearRegression(**best_params)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100  # Convert R² to percentage

    # Close the connection
    conn.close()

    return render_template('evaluation.html', mae=mae, mse=mse, r2=r2, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
