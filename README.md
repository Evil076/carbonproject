# AI Carbon Driven System

## Overview
This project aims to predict and analyze carbon levels based on various features. Users can sign up, log in, and provide input data to get predictions on carbon levels. Additionally, users can leave comments and ratings about their experience.

## Features
- User authentication (signup, login, OTP verification)
- Data prediction based on user input
- Historical data analysis and visualization
- User comments and ratings

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create the database and tables:
   ```bash
   python create.py
   python create_users_table.py
   python comments.py
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the application in your web browser at `http://127.0.0.1:5000`.

## Usage
- Sign up for a new account.
- Log in to access the prediction features.
- Provide input data to get predictions on carbon levels.
- View historical data analysis and trends.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License.
