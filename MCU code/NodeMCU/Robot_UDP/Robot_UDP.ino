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
 *########################################################### */

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

 const char* ssid = "honor";
 const char* password = "password123";
 const char* host = "192.168.43.95";

 char message[2];

 unsigned int localPort = 8080;
 WiFiUDP Udp;

 int move_backward = 15;   ///E2
 int move_forward = 13;
 int turn_left = 5;   // E1
 int turn_right = 4;

 int FBspeed = 500;
 int speed_delta = 50;
 
int FW=0,BK=1,ST=2;
int dir=FW;

 void setup() {
   Serial.begin(57600);
   Serial.print("Started");
   pinMode(move_backward , OUTPUT);
   pinMode(move_forward, OUTPUT);
   pinMode(turn_left, OUTPUT);
   pinMode(turn_right, OUTPUT);

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
   analogWrite(move_backward, LOW);
   analogWrite(move_forward, FBspeed);
   Serial.print(FBspeed);
   
   
 }


 void back()
 {
   delay(10);
   Serial.println("back");
   dir=BK;
   analogWrite(move_backward, FBspeed);
   analogWrite(move_forward, LOW);
   
   //digitalWrite(turn_left, HIGH);
   //digitalWrite(turn_right, LOW);
 }

 void left()
 {
  delay(10);
  Serial.println("left");
   
  digitalWrite(turn_left, HIGH);
  digitalWrite(turn_right, LOW);
 }
 void right()
 {
  delay(10);
  Serial.println("Right");
   
   
   digitalWrite(turn_left, LOW);
   digitalWrite(turn_right, HIGH);
 }


void turnR()
 {
  delay(10);
  Serial.println("Turn0");
   digitalWrite(move_backward, LOW);
   analogWrite(move_forward, FBspeed);
   
   digitalWrite(turn_left, LOW);
   digitalWrite(turn_right, HIGH);
 }
 void turnL()
 {
  delay(10);
  Serial.println("Turn1");
   analogWrite(move_backward, LOW);
  analogWrite(move_forward, FBspeed);
   
   digitalWrite(turn_left, HIGH);
   digitalWrite(turn_right, LOW);
 }

 void turnRB()
 {
  delay(10);
  Serial.println("Turn0B");
   analogWrite(move_backward, FBspeed);
   analogWrite(move_forward, LOW);
   
   digitalWrite(turn_left, LOW);
   digitalWrite(turn_right, HIGH);
 }
 void turnLB()
 {
  delay(10);
  Serial.println("Turn1B");
   analogWrite(move_backward, FBspeed);
  analogWrite(move_forward, LOW);
   
   digitalWrite(turn_left, HIGH);
   digitalWrite(turn_right, LOW);
 }

 void Stop()
 {
  dir=ST;
  delay(10);
  Serial.println("Stop");
  
  analogWrite(move_backward, LOW);
  analogWrite(move_forward, LOW);
   
   digitalWrite(turn_left, LOW);
   digitalWrite(turn_right, LOW);
  
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
if (FBspeed<1000)
   FBspeed = FBspeed + speed_delta;
   Serial.println("increase speed");
   Serial.print(FBspeed);
  }

  void decreaseSpeed(){
    if (FBspeed >0)
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
    case 5:  Stop();                              break;

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

