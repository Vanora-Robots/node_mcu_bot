/* ############################################################ 
 *  By: Vittaysak Rujivorakul
 *  Android App for Controller:  https://play.google.com/store/apps/details?id=com.br3.udpctl&hl=th
 *  My Demo Clip VDO: https://www.youtube.com/watch?v=E85RfNlRmHU 
 *  Hardware Micro Controller: NodeMCU V2 (ESP8266-12) found at http://www.nodemcu.com/
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
 

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

#include <SoftwareSerial.h>
#include <SabertoothSimplified.h>

SoftwareSerial SWSerial(4, 5); // RX on no pin (unused), TX on pin 5   (to S1).
SabertoothSimplified ST(SWSerial); // Use SWSerial as the serial port.

 const char* ssid = "honor";
 const char* password = "password123";
 const char* host = "192.168.43.95";

 char message[2];

 unsigned int localPort = 8080;
 WiFiUDP Udp;

 int righta1 = 15;   ///E2
 int righta2 = 13;
 int lefta1 = 5;   // E1
 int lefta2 = 4;

 int FBspeed = 40;
 int speed_delta = 5;
 
int FW=0,BK=1,STop=2;
int dir=FW;

 void setup() {
   Serial.begin(57600);
   SWSerial.begin(9600);
   Serial.print("Started");
   pinMode(righta1 , OUTPUT);
   pinMode(righta2, OUTPUT);
   pinMode(lefta1, OUTPUT);
   pinMode(lefta2, OUTPUT);

   WiFi.begin(ssid, password);
   while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
   }

   
   Serial.println("");
   Serial.println("WiFi connected");
   Serial.println("IP address: ");
   Serial.println(WiFi.localIP());
   Serial.println("Mac address: ");
   Serial.println(WiFi.macAddress());
   
   Udp.begin(localPort);
   
   Stop();
   delay(10);
   Serial.println("Setup done");
   delay(2000);

 }

 void loop(){
   int packetSize = Udp.parsePacket();
   if (packetSize)
   {
     IPAddress remoteIp = Udp.remoteIP();
     int len = Udp.read(message, 255);
     Serial.println(message);
     checkcommand();
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
   Serial.print(FBspeed);
   
   
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
   analogWrite(righta1, FBspeed);
   analogWrite(righta2, LOW);
   
   digitalWrite(lefta1, LOW);
   digitalWrite(lefta2, HIGH);
 }
 void turnLB()
 {
  delay(10);
  Serial.println("Turn1B");
   analogWrite(righta1, FBspeed);
  analogWrite(righta2, LOW);
   
   digitalWrite(lefta1, HIGH);
   digitalWrite(lefta2, LOW);
 }

 void Stop()
 {
  dir=STop;
  delay(10);
  Serial.println("Stop");
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

 void checkcommand()
 {
   //Control DC Motor
   int cm = atoi(message);
   
   switch(cm){
    //UP
    case 1:                                       break;
    
    //DOWN
    case 2:                                       break;

    //LEFT-90
    case 3:  left();       break;

    //RIGH-90
    case 4:  right();          break;

    //STOP
    case 5:  Stop();                                 break;

   //U-Tern 180
    case 6: turnL();   next();       break;

   //Turn-Left
    case 7: turnL();                              break;

   //Turn-Righ
    case 8:   turnR();                            break;

   //FW
    case 9:   forward();                          break;

    //STEP-FW
    case 10:  forward();   Stop();     break;

   //STEP-LEFT
    case 11: left();    Stop();         break;

   //STEP-RIGH
    case 12: right(); Stop();          break;

   //BACK
    case 13: back();                              break;

   //STEP-BACK
    case 14: back();  Stop();           break;

   //LEFT backward
    case 15: turnLB();                            break;

   //RIGHT backward
    case 16: turnRB();                              break;

    case 17: decreaseSpeed(); next();                     break;

    case 18: increaseSpeed();  next();                    break;

    default: Stop();                              break;
    
   }

 }

