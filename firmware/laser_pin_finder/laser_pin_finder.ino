// Bench tool: find which pin switches the laser transistor.
// Drives one pin HIGH at a time for 3 s and prints its name at 115200 baud.
// Watch the laser: it lights on the matching pin (active-high), or goes dark
// on it if the circuit is active-low. Skips 0/1 (USB serial) and 9/10 (servos).

const int PINS[] = {2, 3, 4, 5, 6, 7, 8, 11, 12, 13, A0, A1, A2, A3, A4, A5};
const char* NAMES[] = {"D2", "D3", "D4", "D5", "D6", "D7", "D8", "D11", "D12", "D13",
                       "A0", "A1", "A2", "A3", "A4", "A5"};
const int PIN_COUNT = sizeof(PINS) / sizeof(PINS[0]);
const unsigned long ON_MS = 3000;
const unsigned long GAP_MS = 1000;

void setup() {
  for (int i = 0; i < PIN_COUNT; i++) {
    digitalWrite(PINS[i], LOW);
    pinMode(PINS[i], OUTPUT);
  }
  Serial.begin(115200);
  Serial.println(F("Laser pin finder. Aim the laser at a wall, not at anyone."));
}

void loop() {
  for (int i = 0; i < PIN_COUNT; i++) {
    Serial.print(F("HIGH: "));
    Serial.println(NAMES[i]);
    digitalWrite(PINS[i], HIGH);
    delay(ON_MS);
    digitalWrite(PINS[i], LOW);
    delay(GAP_MS);
  }
  Serial.println(F("--- cycle done, repeating ---"));
}
