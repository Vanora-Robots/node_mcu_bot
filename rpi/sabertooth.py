#!/usr/bin/python3

import webiopi
from webiopi.devices.serial import Serial
from time import sleep

serial = Serial('ttyACM0', 9600) #enter correct port for serial

def main():
    # write data to serial port

    for i in range (1, 5):
        serial.writeString("M1: 1023\r\n")
        print("M1 Fwd")
        sleep(2)
        serial.writeString("M1: 0\r\n")
        print("M1 Off")
        sleep(2)
        serial.writeString("M2: 1023\r\n")
        print("M2 Fwd")
        sleep(2)
        serial.writeString("M2: 0\r\n")
        print("M2 Off")
        sleep(2)
        serial.writeString("M1: -1023\r\n")
        print("M1 Reverse")
        sleep(2)
        serial.writeString("M1: 0\r\n")
        print("M1 Off")
        sleep(2)
        serial.writeString("M2: -1023\r\n")
        print("M2 Reverse")
        sleep(2)
        serial.writeString("M2: 0\r\n")
        print("M2 Off")

        if (serial.available() > 0):
            data = serial.readString()
            print(data)

        webiopi.sleep(1)

main()
