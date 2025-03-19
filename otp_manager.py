import os
import json

OTP_FILE = 'otps.json'
MAX_OTPS = 5

def load_otps():
    if os.path.exists(OTP_FILE):
        with open(OTP_FILE, 'r') as file:
            return json.load(file)
    return []

def save_otps(otps):
    with open(OTP_FILE, 'w') as file:
        json.dump(otps, file)

def add_otp(otp):
    otps = load_otps()
    otps.append(otp)
    if len(otps) > MAX_OTPS:
        otps.pop(0)  # Remove the oldest OTP
    save_otps(otps)

def get_otps():
    return load_otps()