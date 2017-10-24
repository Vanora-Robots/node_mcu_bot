#!/usr/bin/env python

from __future__ import print_function

import pygame
import serial
import sys

SABERTOOTH_PORT_NAME = '/dev/ttyACM0'
SABERTOOTH_PORT_BAUDRATE = 9600
SABERTOOTH_PORT_BYTESIZE = 8
sabertooth = None

try:
    sabertooth = serial.Serial(port=SABERTOOTH_PORT_NAME,
                             baudrate=SABERTOOTH_PORT_BAUDRATE,
                             bytesize=SABERTOOTH_PORT_BYTESIZE,
                             parity=serial.PARITY_NONE,
                             writeTimeout=0,
                             stopbits=serial.STOPBITS_ONE,
                             dsrdtr=True)
except:
    print ("Failed to bring in the Sabertooth")


def main():
  try:
    prev_cmd = cmd = chr(0)
    keys_pressed = pygame.event.get()  # pygame.event.get(pygame.KEYDOWN)
    # print(keys_pressed)
    keys_pressed_list = [0,0,0,0,0]
    for event in keys_pressed:
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
            sabertooth.drive(1, 127)
            sabertooth.drive(2, 127)
        if event.key == pygame.K_DOWN:
            sabertooth.drive(1, -127)
            sabertooth.drive(2, -127)
        if event.key == pygame.K_LEFT:
            sabertooth.drive(1, -127)
            sabertooth.drive(2, 127)
        if event.key == pygame.K_RIGHT:
            sabertooth.drive(1, 127)
            sabertooth.drive(2, -127)
        if event.key == pygame.K_SPACE:
            sabertooth.drive(1, 0)
            sabertooth.drive(2, 0)
        if event.key == pygame.K_KP_PLUS:
          keys_pressed_list.append("+")
        if event.key == pygame.K_KP_MINUS:
          keys_pressed_list.append("-")
        if event.key == pygame.K_q:
            exit()

      if prev_cmd != cmd:
        sabertooth.write(cmd)
        sabertooth.flush()
        prev_cmd = cmd
  except KeyboardInterrupt:
    print "Sabertooth conn closed"
    sabertooth.close()
  return 0

if __name__ == '__main__':
    if sabertooth is None:
        print('Sabertooth not found.')
        exit()
    print('Initialized sabertooth.')
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    sabertooth.
    while True:
        main()
