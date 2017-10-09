## NodeBot

A WiFi enabled robot with Android and NodeMCU connectivity .

![Alt text](screenshots/.screenshot.png?raw=true "Example Image")

In order to use NodeBot, clone this repository.

### Demo

The package contains a launch file for demonstration purposes. Use it to verify your installation and to get started:

``python scripts/start.py -d 0``

###Debug

To Install all the required packages

``sudo pip install -r requirments.txt``

-d 0 , corresponds to /dev/video0. To find out the device address type below

``ls /dev/video*``

Arduino IDE needs to be 1.6xx or greater .(ESP8266/ESP32 support is there officially)

### Usage

Connect NodeMcu to Arduino IDE , Set the correct Pin Configuration and Upload the code.


### Options

- `start video`, This will start a video feed from the video capture device connected.
- `Tracker` , You will get a pop up screen asking you to select an object in the frame to follow.
- `Calibrate ` , This will help calibrate the camera using a Checkerboard.
- `Test Calibrate` , This is a module which will give QR code pose estimation .
- `CNN Face detector`  , This module is still under development


### Questions, Bugs

Contact the author (https://github.com/chrissunny94 on github), or open an issue.
