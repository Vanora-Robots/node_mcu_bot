/* ############################################################ 
 *  By: Chris Sunny Thaliyath
 *  
 *  
 *  NodeMCU has weird pin mapping.
Pin numbers written on the board itself do not correspond to ESP8266 GPIO pin numbers. We have constants defined to make using this board easier:

static const uint8_t D0   = 16;
static const uint8_t D1   = 5;
static const uint8_t D2   = 4;
static const uint8_t D3   = 0;
static const uint8_t D4   = 2;
static const uint8_t D5   = 14;
static const uint8_t D6   = 12;
static const uint8_t D7   = 13;
static const uint8_t D8   = 15;
static const uint8_t D9   = 3;
static const uint8_t D10  = 1;
 *  Enjoy your kids
 *  
 *  
 *  DIP switch settings on Sabertooth
 *  1 - OFF
 *  2 - OFF
 *  3 - ON 
 *  4 - OFF
 *  5 - ON
 *  6 - ON
 *########################################################### */
 


#include <SoftwareSerial.h>
#include <SabertoothSimplified.h>

SoftwareSerial SWSerial(2, 3); // RX on no pin (unused), TX on pin 3   (to S1).
SabertoothSimplified ST(SWSerial); // Use SWSerial as the serial port.

 char message[2];


 int FBspeed = 40;
 int speed_delta = 5;
 
int FW=0,BK=1,STop=2;
int dir=FW;

 void setup() {
   Serial.begin(9600);
   SWSerial.begin(9600);
   Serial.print("Started");
   

   ST.motor(1, FBspeed);
   ST.motor(2, FBspeed);
   
   Stop();
   delay(10);
   Serial.println("Setup done");
   delay(2000);

 }

 void loop(){
  
   
   char inByte = ' ';
   if (Serial.available())
   {
     char inByte = Serial.read();
     Serial.println(inByte);
     checkcommand(inByte);
     message[0]=0;
     message[1]=0;
     delay(5);
   }
} 
 void forward()
 {
   delay(10);
   Serial.println("forward");
   
   dir=FW;
   ST.motor(1, FBspeed);
   ST.motor(2, FBspeed);
   
   
   
 }


 void back()
 {
   delay(10);
   Serial.println("back");
   dir=BK;
   ST.motor(1, -FBspeed);
   ST.motor(2, -FBspeed);
   
   //digitalWrite(lefta1, HIGH);
   //digitalWrite(lefta2, LOW);
 }

 void left()
 {
  delay(10);
  Serial.println("left");
  ST.motor(1, -FBspeed);
  ST.motor(2, FBspeed); 
 }
 void right()
 {
  delay(10);
  Serial.println("Right");
  ST.motor(1, FBspeed);
   ST.motor(2, -FBspeed);
 }


void turnR()
 {
  delay(10);
  Serial.println("Turn0");
   ST.motor(1, 127);
   ST.motor(2, -FBspeed);
 }
 void turnL()
 {
  delay(10);
  Serial.println("Turn1");
   ST.motor(1, -FBspeed);
   ST.motor(2, 127);
 }

 void turnRB()
 {
  delay(10);
  Serial.println("Turn0B");
   
 }
 void turnLB()
 {
  delay(10);
  Serial.println("Turn1B");
   
 }

 void Stop()
 {
  dir=STop;
  delay(10);
  //Serial.println("Stop");
  ST.motor(1, 0);
  ST.motor(2, 0);
 }

 void next(){
            if(dir==FW){
              forward();
            }else if(dir==BK){
               back();
            }else{
              Stop(); 
            }
 }




 

 void increaseSpeed(){
if (FBspeed<127)
   FBspeed = FBspeed + speed_delta;
   Serial.println("increase speed");
   Serial.print(FBspeed);
  }

  void decreaseSpeed(){
    if (FBspeed >-127)
   FBspeed = FBspeed - speed_delta;
   Serial.println("decreased  speed");
   Serial.print(FBspeed);
  
  }


 
 
 
 void checkcommand(char cm)
 {
   
   
   switch(cm){
    //UP
    case 1:                                       break;
    
    //DOWN
    case 'D':                                       break;

    //LEFT-90
    case 'L':  left();       break;

    //RIGH-90
    case 'R':  right();          break;

    //STOP
    case 'S':  Stop();                                 break;

   //U-Tern 180
    case 6: turnL();   next();       break;

   //Turn-Left
    case 'l': turnL();                              break;

   //Turn-Righ
    case 'r':   turnR();                            break;

   //FW
    case 'F':   forward();                          break;

   
   //STEP-LEFT
    case 'q': left();    Stop();         break;

   //STEP-RIGH
    case 'e': right(); Stop();          break;

   //BACK
    case 'B': back();                              break;

   
  
    case 'k': decreaseSpeed(); next();                     break;

    case 'i': increaseSpeed();  next();                    break;

    default: Stop();                              break;
    
   }

 }

