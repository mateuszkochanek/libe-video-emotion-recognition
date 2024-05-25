# Ran on raspberry pi

import socket
import RPi.GPIO as GPIO

# Set up the GPIO pin for the LED
# LED_PIN = 18
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(LED_PIN, GPIO.OUT)

# Set up the server
HOST = ''  # Accept connections from any address
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Server is listening...")
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            command = data.decode('utf-8')
            if command == "1":
                print("YOU ARE NOT HAPPYYYY!!!!")
                # GPIO.output(LED_PIN, GPIO.HIGH)
            elif command == "0":
                print("Good :)")
                # GPIO.output(LED_PIN, GPIO.LOW)
            else:
                print("Invalid command received")

# GPIO.cleanup()