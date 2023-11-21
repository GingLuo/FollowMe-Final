# Don't forget to do this:
# sudo pip install Jetson.GPIO

import Jetson.GPIO as GPIO
import time
import pyttsx3
import sys
import signal
# Pin Definitions
input_pin = 31  # BCM pin 18, BOARD pin 12

def low_bat(something):
    i = 0
    while True:
        value = GPIO.input(input_pin)
        if (value == GPIO.HIGH):
            engine = pyttsx3.init('espeak')
            engine.setProperty('volumn', 0.7)
            engine.say("Low Battery")
            print("low-battery + %d", i)
            i += 1
            engine.runAndWait()
        else:
            return

def signal_handler():
    GPIO.cleanup()
    sys.exit(0)

def main():
    prev_value = None
    
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin
    GPIO.add_event_detect(input_pin, GPIO.RISING, callback=low_bat, bouncetime=10)
    print("Starting demo now! Press CTRL+C to exit")
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()

if __name__ == '__main__':
    main()
