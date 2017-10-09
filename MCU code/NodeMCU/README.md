# NodeMCU_Robot
First version for Home DIY NodeMCU (ESP8266-12E) Smart Care Robot, It just $30 and made it your self.

    We can use Any Dual H-Bridge Motor Controller
- l293
- l298
- sabertooth  


Please change the below settings

``const char* ssid = "honor";``

``const char* password = "password123";``

``const char* host = "192.168.1.3";`` - Ip address of your PC


ps:- I will write a script soon which will do the flashing automatically (Setting the IP, username and password correctly)

https://github.com/TomFreudenberg/udoo-arduino-cli , i have included the .deb file i have used to automate the upload form the CLI.

``sudo chmod +x settings.sh``

``sudo ./settings.sh``

![Alt text](screenshots/.img1.jpg?raw=true "Example Image")

Example VDO Clip:
https://www.youtube.com/watch?v=E85RfNlRmHU

![Alt text](screenshots/.img2.jpg?raw=true "Example Image")

