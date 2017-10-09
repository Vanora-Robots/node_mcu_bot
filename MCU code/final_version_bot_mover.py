from pynput.keyboard import Key, Listener
from copy import copy
import socket
import time

#Network packet specs
UDP_IP = "192.168.43.82"
UDP_PORT = 8080
MESSAGE = "Hello, World!"
print "UDP target IP:", UDP_IP
print "UDP target port:", UDP_PORT
print "message:", MESSAGE
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#Key status variables
key_status = []
pre_key_status = []
key_list = ['Key.up','Key.down','Key.left','Key.right','Key.space']
for i in range(0,len(key_list)):
    key_status.append(0)
    pre_key_status.append(9)

speed_status_up = 0
speed_status_down = 0

    
#Function changes the key_status variable active/deactive 1/0
def key_status_toggle(i,status):
    global key_status
    if status == 0:
        if key_status[i] == 0:
            key_status[i] = 1
    else:
        if key_status[i] == 1:
            key_status[i] = 0

def key_response(key_status):
    #print key_status
    #stop
    if key_status == [0,0,0,0,0]:
        return 0
    #move forward
    elif key_status == [1,0,0,0,0]:
        return 9
    #move backward
    elif key_status == [0,1,0,0,0]:
        return 13


    #move left forward
    elif key_status == [1, 0, 1, 0, 0]:
        return 7
    # move left backward
    elif key_status == [0, 1, 1, 0, 0]:
        return 15
    #left turn 90deg
    elif key_status == [0, 0, 1, 0, 0]:
        return 3



    #move right forward
    elif key_status == [1,0 , 0, 1, 0]:
        return 8
    # move right backward
    elif key_status == [0, 1, 0, 1, 0]:
        return 16
    # right turn 90deg
    elif key_status == [0, 0, 0, 1, 0]:
        return 4


    else :
        return 0
	



            
#Function Checks the key and decides to send the status to the network or not
def key_status_change(key,status):
    global key_status
    global pre_key_status
    global key_list

    
    key_to_str = str(key)

    if key_to_str in key_list:
        key_status_toggle(key_list.index(key_to_str),status)
        
        if pre_key_status != key_status:
                
            sock.sendto(str(key_response(key_status)), (UDP_IP, UDP_PORT))
            pre_key_status=copy(key_status)
            print key_status

                
        else:
                
            print "Key status already sent"

    else:
        print "Please press: UP - Forward , DOWN - Reverse , LEFT - Turn Left , RIGHT - Turn Right, SPACE - Stop"
        
    
def speed_status_change(up_down):
    sock.sendto(str(up_down), (UDP_IP, UDP_PORT))
    print up_down

    



def on_press(key):
   
    global speed_status_up
    global speed_status_down
    try:
        if key.char =='-':
            speed_status_down=1
        elif key.char == '+':
            speed_status_up = 1
        else:
            key_status_change(key, 0)

    except AttributeError:
        key_status_change(key, 0)


    
    
    
    
    
def on_release(key):
    global speed_status_up
    global speed_status_down
    try:
        if key.char == '-':
            if speed_status_down == 1:
                speed_status_change(17)
                speed_status_down = 0
            else:
                print "- is not pressed"

        elif key.char == '+':
            if speed_status_up == 1:
                speed_status_change(18)
                speed_status_up = 0
            else:
                print "+ is not pressed"

        else:
            key_status_change(key, 1)

    except AttributeError:
        key_status_change(key, 1)
    

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
