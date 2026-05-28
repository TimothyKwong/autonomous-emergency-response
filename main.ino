#include <SPI.h>                              // SPI library
#include <Adafruit_Sensor.h>                  // Core sensor library
#include <Adafruit_ST7789.h>                  // Hardware-specific library for ST7789
#include <Adafruit_ADS1X15.h>                 // Hardware-specific library for ADS1x15
#include "Adafruit_BME680.h"                  // Hardware-specific library for BME680
#include <Adafruit_AHTX0.h>                   // Hardware-specific library for AHT20 - Connect external AHT20 with provided wire to Port J7
#include <Fonts/FreeSerif9pt7b.h>             // Font to be used on the display
#include "SparkFun_BMA400_Arduino_Library.h"  // Hardware-specific library for BMA400
#include <RCWL_1X05.h>                        // Hardware-specific library for Ultrasonic Sensor

// Pin Definitions
#define TFT_CS 38
#define TFT_RST -1
#define TFT_DC 48

// Other Constants
#define LED_PIN LED_BUILTIN
#define BME_Addr 0x76
#define SEALEVELPRESSURE_HPA (1013.25)

uint32_t t_prev = 0;
size_t sensorDataSize = 0;

SPIClass* fspi = NULL;
Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

Adafruit_BME680 bme;
Adafruit_ADS1015 ads;
RCWL_1X05 ultrasonic;

#pragma pack(push, 1)
struct SensorResults 
{
    float P=0.0f;
    float T=0.0f;
    float RH=0.0f;

    int16_t Light=0;
    float vLight=0.0f;

    float USDistance=0.0f;
    float speedOfSound=0.0f;

    float loop_time_ms=0.0f;
} SensorData;
#pragma pack(pop)

void setup() 
{
    Serial.begin(115200);

    // Turn off LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Init Display
    fspi = new SPIClass(FSPI);
    fspi->begin(36, 37, 35);
    tft.init(135, 240);
    tft.setRotation(3);

    // Init all sensors
    Wire.begin(16, 15);   //I2C Bus #1 Init (For BME, BMA, and ADC)
    Wire1.begin(39, 40);  //I2C Bus #2 Init (For Ultrasonic)

    // Init BME680
    if (!bme.begin(BME_Addr)) {
        Serial.println("Could not find a valid BME680 sensor, check wiring!");
    }

    // Set up oversampling and filter for BME680 - sensor no longer in use
    bme.setTemperatureOversampling(BME680_OS_1X);
    bme.setHumidityOversampling(BME680_OS_1X);
    bme.setPressureOversampling(BME680_OS_1X);
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    // bme.setGasHeater(320, 150);

    // Init ADC (ADS1015)
    if (!ads.begin()) {
        Serial.println("Failed to initialize ADS.");
    }

    // Init Ultrasonic Sensor
    if (!ultrasonic.begin(&Wire1)) {
        Serial.println("Sensor not found.");
    } else {
        ultrasonic.setMode(RCWL_1X05::oneShot); 
    }
}

void loop() 
{
    // Track loop time
    uint32_t t_now = micros();
    SensorData.loop_time_ms = (t_now - t_prev) / 1000.0f;
    t_prev = t_now;

    // Main loop
    readSensors();   // 10ms
    printSensors();  // 0ms
    updateDisplay(); // 40ms
    delay(50);       // 50ms
}

void readSensors() 
{
    // Photoresistance for display
    SensorData.Light = ads.readADC_SingleEnded(2);
    SensorData.vLight = ads.computeVolts(SensorData.Light);
}

void printSensors() 
{
    sensorDataSize = sizeof(SensorData);
    Serial.write((uint8_t*)&SensorData, sizeof(SensorData));
}

void updateDisplay() 
{
    const int COL1_X = 0;
    const int COL2_X = 120;

    const uint16_t TITLE_COLOR = ST77XX_WHITE;
    const uint16_t VALUE_COLOR = ST77XX_CYAN;

    tft.fillScreen(ST77XX_BLACK);
    tft.setTextWrap(false);
    tft.setFont(&FreeSerif9pt7b);

    tft.setTextColor(TITLE_COLOR);
    tft.setCursor(COL1_X, 113);
    tft.print("Ultrasonic\n");

    tft.setTextColor(VALUE_COLOR);
    tft.setCursor(COL1_X, 32);
    tft.print("T: "); tft.print(SensorData.T, 2); tft.println(" °C");

    tft.setCursor(COL1_X, 48);
    tft.print("H: "); tft.print(SensorData.RH, 2); tft.println(" %");

    tft.setCursor(COL1_X, 64);
    tft.print("P: "); tft.print(SensorData.P, 1); tft.println(" kPa");

    tft.setCursor(COL1_X, 128);
    tft.print(SensorData.USDistance, 0); tft.println(" mm");

    tft.setTextColor(VALUE_COLOR);
    tft.setCursor(COL2_X, 48);
    tft.print("L: "); tft.print(SensorData.vLight, 3); tft.println("V");

    tft.setCursor(COL2_X, 64);
    tft.print("Time: "); tft.print(SensorData.loop_time_ms); tft.println("ms");

    tft.setCursor(COL2_X, 128);
    tft.print("Size: "); tft.print(sensorDataSize); tft.println("bytes");
}