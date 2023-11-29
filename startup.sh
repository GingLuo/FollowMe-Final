#! /bin/bash

kill_process(){
	echo "Killing Process..."
	kill $1
	sleep 1
	exit 0
}
pactl set-default-sink alsa_output.usb-Jieli_Technology_UACDemoV1.0_1120040804060316-00.analog-stereo
pactl set-sink-volume @DEFAULT_SINK@ 80%
python3 /home/followme/Desktop/FollowMe-Final/newinference.py &

THE_PID=$!

trap "kill_process $THE_PID" SIGINT

while [ 1 -eq 1 ]; do
	sleep 30 &
	wait
done
