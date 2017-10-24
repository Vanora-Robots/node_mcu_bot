#define echoPin 8 // Echo Pin
#define trigPin 9 // Trigger Pin
#define LEDPin 13 // Onboard LED

int maximumRange = 200; // Maximum range needed
int minimumRange = 0; // Minimum range needed
long duration, distance; // Duration used to calculate distance

void setup() {
 Serial.begin (9600);
 pinMode(trigPin, OUTPUT);
 pinMode(echoPin, INPUT);
 pinMode(LEDPin, OUTPUT); // Use LED indicator (if required)
 pinMode(6, OUTPUT);
 pinMode(7, INPUT);
 
}

float ultrasonic_distance(int echo , int trigger){
  
  /* The following trigPin/echoPin cycle is used to determine the
 distance of the nearest object by bouncing soundwaves off of it. */ 
 digitalWrite(trigger, LOW); 
 delayMicroseconds(2); 

 digitalWrite(trigger, HIGH);
 delayMicroseconds(10); 
 
 digitalWrite(trigger, LOW);
 duration = pulseIn(echo, HIGH);
 
 //Calculate the distance (in cm) based on the speed of sound.
 distance = duration/58.2;
  
  
  return distance;
  }

void loop() {

Serial.print(ultrasonic_distance(8,9));
delay(50);
Serial.print(" , ");
Serial.print(ultrasonic_distance(7,6));
Serial.print(" , ");
Serial.print(analogRead(A1));
Serial.println(); 
delay(50);
}
