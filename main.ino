#include <HardwareSerial.h>
#include <TinyGPS++.h>

// GPS Module
#define GPS_RX 16
#define GPS_TX 17
HardwareSerial gpsSerial(2); // Use UART2 for GPS
TinyGPSPlus gps;

// MQ3 Sensor
#define MQ3_PIN 34

void setup() {
  // Initialize Serial Monitor (USB)
  Serial.begin(115200);

  // Initialize GPS Serial
  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);

  // Initialize MQ3 Sensor
  pinMode(MQ3_PIN, INPUT);
}

void loop() {
  // Read GPS Data
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  if (gps.location.isUpdated()) {
    // Send GPS Data to Laptop
    Serial.print("GPS:");
    Serial.print(gps.location.lat(), 6);
    Serial.print(",");
    Serial.print(gps.location.lng(), 6);
    Serial.print(",");
    Serial.println(gps.speed.kmph());
  }

  // Read MQ3 Sensor
  int mq3Value = analogRead(MQ3_PIN);

  // Send MQ3 Data to Laptop
  Serial.print("MQ3:");
  Serial.println(mq3Value);

  delay(1000); // Adjust delay as needed
}