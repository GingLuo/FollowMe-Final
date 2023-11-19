sudo groupadd -f -r gpio
sudo usermod -a -G gpio followme

sudo pip install Jetson.GPIO

sudo cp venv/lib/pythonNN/site-packages/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/

sudo udevadm control --reload-rules && sudo udevadm trigger