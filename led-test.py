import RPi.GPIO as GPIO
import time

# Set up the GPIO pin numbering
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

try:
    while True:
        GPIO.output(17, GPIO.HIGH)  # Turn on the LED
        time.sleep(1)               # Wait for 1 second
        time.sleep(1)               # Wait for 1 second
except KeyboardInterrupt:
    pass
finally:
    GPIO.output(17,GPIO.LOW)
    GPIO.cleanup()  # Clean up the GPIO pins to their default state