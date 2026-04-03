#include <DHT.h>
#include <Servo.h>

// ----------------- Hall sensor KY-003 -----------------
const int hallPin = 6;
int hallState = 0;

// ----------------- Ultrasonic sensor HC-SR04 -----------------
const int trigPin = 3;
const int echoPin = 4;
long duration;
int distance;

// ----------------- Servo -----------------
const int servoPin = 5;
Servo myServo;

// ----------------- DHT11 sensor -----------------
#define DHTPIN 2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// ----------------- Setup -----------------
void setup() {
  pinMode(hallPin, INPUT);        // Hall sensor
  pinMode(trigPin, OUTPUT);       // HC-SR04
  pinMode(echoPin, INPUT);

  dht.begin();                    // DHT11

  myServo.attach(servoPin);
  myServo.write(0);               // initial servo position

  Serial.begin(9600);             // Serial Monitor
}

// ----------------- Main loop -----------------
void loop() {
  // ----------------- Hall sensor -----------------
  hallState = digitalRead(hallPin);
  if (hallState == LOW) {
    Serial.println("Magnetic field detected");
  } else {
    Serial.println("Clear space");
  }

  // ----------------- Ultrasonic sensor with scaling -----------------
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;  // real distance in cm

  // Scale: 10 cm → 120 km
  float scaledDistance = distance * 120.0 / 10.0; // km
  Serial.print("Distance to object : ");
  Serial.print(scaledDistance, 1);  // one decimal
  Serial.println(" km");

  // ----------------- Servo control based on real distance -----------------
  if (distance > 0 && distance < 10) {
    myServo.write(90);
  } else {
    myServo.write(0);
  }

  // ----------------- Temperature & Humidity -----------------
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (isnan(h) || isnan(t)) {
    Serial.println("Sensor error");
  } else {
    Serial.print("Temperature: "); Serial.print(t); Serial.print(" °C, ");
    Serial.print("Humidity: "); Serial.print(h); Serial.println(" %");
  }

  Serial.println("---------------");
  delay(200); // small pause for loop
}