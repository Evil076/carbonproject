import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_otp(email, otp):
    sender_email = "carbon.2025M@gmail.com"
    sender_password = "@Mbagaran006*"  # Use the correct password

    receiver_email = email

    message = MIMEMultipart("alternative")
    message["Subject"] = "Your OTP Code"
    message["From"] = sender_email
    message["To"] = receiver_email

    text = f"Your OTP code is {otp}"
    part = MIMEText(text, "plain")
    message.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        logger.info(f"OTP sent to {receiver_email}: {otp}")
        print("Email sent successfully")
    except Exception as e:
        logger.error(f"Failed to send email to {receiver_email}: {e}")
        print(f"Failed to send email: {e}")
