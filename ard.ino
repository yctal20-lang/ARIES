#include <DHT.h>
#include <Servo.h>


const int hallPin = 6;
int hallState = 0;


const int trigPin = 3;
const int echoPin = 4;
long duration;
int distance;


const int servoPin = 5;
Servo myServo;


#define DHTPIN 2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);


void setup() {
  pinMode(hallPin, INPUT);        
  pinMode(trigPin, OUTPUT);       
  pinMode(echoPin, INPUT);

  dht.begin();                    

  myServo.attach(servoPin);
  myServo.write(0);               
  Serial.begin(9600);             
}


void loop() {
  
  hallState = digitalRead(hallPin);
  if (hallState == LOW) {
    Serial.println("Magnetic field detected");
  } else {
    Serial.println("Clear space");
  }

  
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;  

  
  float scaledDistance = distance * 120.0 / 10.0; 
  Serial.print("Distance to object : ");
  Serial.print(scaledDistance, 1);  
  Serial.println(" km");

  
  if (distance > 0 && distance < 10) {
    myServo.write(90);
  } else {
    myServo.write(0);
  }

  
  float t = dht.readTemperature();
  if (isnan(h) || isnan(t)) {
    Serial.println("Sensor error");
  } else {
    Serial.print("Temperature: "); Serial.print(t); Serial.print(" °C, ");
    Serial.print("Humidity: "); Serial.print(h); Serial.println(" %");
  }

  Serial.println("---------------");
  delay(200); 
}