from flask import Flask, request
import RPi.GPIO as GPIO
import time
import threading

app = Flask(__name__)

# Set up GPIO
RELAY_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)

# Global variable to track the cooldown period
cooldown_end_time = 0
lock = threading.Lock()

def destroy():
    GPIO.output(RELAY_PIN, GPIO.LOW)
    GPIO.cleanup()

def set_relay_low_after_delay():
    time.sleep(0.2)
    GPIO.output(RELAY_PIN, GPIO.LOW)

@app.route('/led', methods=['POST'])
def control_led():
    global cooldown_end_time

    state = request.form.get('state')
    current_time = time.time()

    if state == '1':
        with lock:
            if current_time > cooldown_end_time:
                cooldown_end_time = current_time + 5  # Set cooldown period for next 5 seconds
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                threading.Thread(target=set_relay_low_after_delay).start()
                return 'LED ON', 200
            else:
                return 'LED ON COOLDOWN', 429  # Return a response indicating the cooldown is active
    elif state == '0':
        #GPIO.output(RELAY_PIN, GPIO.LOW)
        return 'LED OFF', 200
    else:
        return 'Invalid state', 400

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        destroy()
    except Exception as e:
        print(e)
        destroy()
    finally:
        destroy()
