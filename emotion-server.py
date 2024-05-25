from flask import Flask, request
import RPi.GPIO as GPIO

app = Flask(__name__)

# Set up GPIO
#LED_PIN = 17
#PIO.setmode(GPIO.BCM)
#GPIO.setup(LED_PIN, GPIO.OUT)

@app.route('/led', methods=['POST'])
def control_led():
    state = request.form.get('state')
    if state == '1':
        print("ypu are a sad boi, bad!")
        #GPIO.output(LED_PIN, GPIO.HIGH)
        return 'LED ON', 200
    elif state == '0':
        print("ypu are a happy boi, good!")
        #GPIO.output(LED_PIN, GPIO.LOW)
        return 'LED OFF', 200
    else:
        return 'Invalid state', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)