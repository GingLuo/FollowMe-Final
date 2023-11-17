# Don't forget to do this:
# sudo pip install Jetson.GPIO

import Jetson.GPIO as GPIO
import time
import pyttsx3
# Pin Definitions
input_pin = 31  # BCM pin 18, BOARD pin 12

def low_bat():
    while True:
        value = GPIO.input(input_pin)
        if (value == GPIO.HIGH):
            engine = pyttsx3.init('espeak')
            engine.say("Low Battery")
            engine.runAndWait()
        else:
            return

def main():
    prev_value = None
    
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin
    GPIO.add_event_detect(input_pin, GPIO.RISING, callback=low_bat, bouncetime=10, polltime=0.2)
    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
            # value = GPIO.input(input_pin)
            # if value != prev_value:
            #     if value == GPIO.HIGH:
            #         value_str = "HIGH"
            #     else:
            #         value_str = "LOW"
            #     print("Value read from pin {} : {}".format(input_pin,
            #                                                value_str))
            #     prev_value = value
            time.sleep(1)
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    main()