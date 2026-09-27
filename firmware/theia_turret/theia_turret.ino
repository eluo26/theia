// Theia turret firmware: a command-driven pan/tilt servo turret with a laser.
// Protocol (115200 baud, one ASCII line per command, one reply line each):
//   boot            -> READY
//   PING            -> OK
//   HOME            -> OK <pan> <tilt>
//   GOTO <pan> <tilt> -> OK <pan> <tilt> | ERR <reason>   (servo degrees)
//   LASER 1|0       -> OK
//   anything else   -> ERR unknown
// See docs/turret-integration-spec.md section 3.

#include <Servo.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// ---- Pins ------------------------------------------------------------------
const int PAN_PIN = 9;                // large servo
const int TILT_PIN = 10;              // micro servo
const int LASER_PIN = 8;              // found with laser_pin_finder
const bool LASER_ACTIVE_HIGH = true;

// ---- Limits and home (servo degrees; tune at the bench) ---------------------
const int PAN_MIN = 0;
const int PAN_MAX = 180;
const int TILT_MIN = 0;               // 0 = up
const int TILT_MAX = 45;              // 45 = down
const int PAN_HOME = 0;
const int TILT_HOME = 0;              // TODO bench: servo angle where the laser is level

// ---- Move timing -------------------------------------------------------------
const unsigned long PAN_MS_PER_DEG = 8;
const unsigned long TILT_MS_PER_DEG = 4;
const unsigned long MARGIN_MS = 100;

// ---- Laser safety ------------------------------------------------------------
const unsigned long LASER_MAX_ON_MS = 20000;

Servo panServo;
Servo tiltServo;
int panPos = PAN_HOME;
int tiltPos = TILT_HOME;

bool laserOn = false;
unsigned long laserOnAt = 0;

const byte LINE_MAX = 32;
char line[LINE_MAX];
byte lineLen = 0;
bool lineOverflow = false;

void setLaser(bool on) {
  digitalWrite(LASER_PIN, (on == LASER_ACTIVE_HIGH) ? HIGH : LOW);
  laserOn = on;
  if (on) {
    laserOnAt = millis();
  }
}

bool inRange(long pan, long tilt) {
  return pan >= PAN_MIN && pan <= PAN_MAX && tilt >= TILT_MIN && tilt <= TILT_MAX;
}

// Moves both servos and blocks until the estimated travel time has passed.
// Hobby servos give no position feedback, so the time is estimated from distance.
void moveTo(int pan, int tilt) {
  // Never sweep a lit laser across the room.
  setLaser(false);

  unsigned long panMs = (unsigned long)abs(pan - panPos) * PAN_MS_PER_DEG;
  unsigned long tiltMs = (unsigned long)abs(tilt - tiltPos) * TILT_MS_PER_DEG;
  panServo.write(pan);
  tiltServo.write(tilt);
  panPos = pan;
  tiltPos = tilt;
  delay(max(panMs, tiltMs) + MARGIN_MS);
}

void replyPose() {
  Serial.print(F("OK "));
  Serial.print(panPos);
  Serial.print(' ');
  Serial.println(tiltPos);
}

// Parses one integer, skipping leading spaces. Advances p past it.
bool parseLong(const char*& p, long& out) {
  while (*p == ' ') {
    p++;
  }
  char* end;
  out = strtol(p, &end, 10);
  if (end == p) {
    return false;
  }
  p = end;
  return true;
}

void handleGoto(const char* args) {
  long pan;
  long tilt;
  const char* p = args;
  if (!parseLong(p, pan) || !parseLong(p, tilt)) {
    Serial.println(F("ERR bad_args"));
    return;
  }
  while (*p == ' ') {
    p++;
  }
  if (*p != '\0') {
    Serial.println(F("ERR bad_args"));
    return;
  }
  if (!inRange(pan, tilt)) {
    Serial.println(F("ERR out_of_range"));
    return;
  }
  moveTo((int)pan, (int)tilt);
  replyPose();
}

void handleLine(char* cmd) {
  for (char* c = cmd; *c; c++) {
    *c = toupper(*c);
  }

  if (strcmp(cmd, "PING") == 0) {
    Serial.println(F("OK"));
  } else if (strcmp(cmd, "HOME") == 0) {
    moveTo(PAN_HOME, TILT_HOME);
    replyPose();
  } else if (strncmp(cmd, "GOTO ", 5) == 0) {
    handleGoto(cmd + 5);
  } else if (strcmp(cmd, "LASER 1") == 0) {
    setLaser(true);
    Serial.println(F("OK"));
  } else if (strcmp(cmd, "LASER 0") == 0) {
    setLaser(false);
    Serial.println(F("OK"));
  } else {
    Serial.println(F("ERR unknown"));
  }
}

// Non-blocking line reader. Ignores '\r' so "\r\n" line endings work too.
void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      if (lineOverflow) {
        Serial.println(F("ERR too_long"));
      } else if (lineLen > 0) {
        line[lineLen] = '\0';
        handleLine(line);
      }
      lineLen = 0;
      lineOverflow = false;
    } else if (lineLen < LINE_MAX - 1) {
      line[lineLen++] = c;
    } else {
      lineOverflow = true;
    }
  }
}

void setup() {
  // Laser off before anything else: the board resets whenever the port opens.
  digitalWrite(LASER_PIN, LASER_ACTIVE_HIGH ? LOW : HIGH);
  pinMode(LASER_PIN, OUTPUT);
  setLaser(false);

  Serial.begin(115200);

  // Writing before attach makes the first pulse go to home instead of 90.
  panServo.write(PAN_HOME);
  tiltServo.write(TILT_HOME);
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  // Starting position is unknown, so wait for a worst-case full sweep.
  delay((PAN_MAX - PAN_MIN) * PAN_MS_PER_DEG + MARGIN_MS);

  Serial.println(F("READY"));
}

void loop() {
  readSerial();

  if (laserOn && millis() - laserOnAt >= LASER_MAX_ON_MS) {
    setLaser(false);
  }
}
