int pwm_a = 3;  //PWM control for motor outputs 1 and 2 
int pwm_b = 9;  //PWM control for motor outputs 3 and 4 
int dir_a = 2;  //direction control for motor outputs 1 and 2 
int dir_b = 8;  //direction control for motor outputs 3 and 4 

// duration for output
int time = 50;
// initial command
int command = 0;

void setup() {
  pinMode(pwm_a, OUTPUT);  //Set control pins to be outputs
  pinMode(pwm_b, OUTPUT);
  pinMode(dir_a, OUTPUT);
  pinMode(dir_b, OUTPUT);
  Serial.begin(115200);
}

void loop() {
  //receive command
  if (Serial.available() > 0){
    command = Serial.read();
  }
  else{
    reset();
  }
   send_command(command,time);
}

void right(int time){
  // Right
  analogWrite(dir_a, LOW);
  analogWrite(pwm_a, 255); 
  
  delay(time);
}

void left(int time){
 // Left
 digitalWrite(dir_a, HIGH);
  analogWrite(pwm_a, 255); 
  delay(time);
}

void forward(int time){
  // Forward
  digitalWrite(dir_b, HIGH);    
  analogWrite(pwm_b, 100);
  delay(time);
}

void reverse(int time){
// Reverse
  digitalWrite(dir_b, LOW);
  analogWrite(pwm_b, 100);

  delay(time);
}

void forward_right(int time){
// Forward Right
  digitalWrite(dir_b, HIGH);      
  digitalWrite(dir_a, LOW);
  analogWrite(pwm_a, 255); 
  analogWrite(pwm_b, 100);
  delay(time);
}

void reverse_right(int time){
 // Reverse Right
 digitalWrite(dir_b, LOW);
 digitalWrite(dir_a, LOW);
 analogWrite(pwm_b, 100);
 analogWrite(pwm_a, 255); 
  delay(time);
}

void forward_left(int time){
// Foward Left
  digitalWrite(dir_b, HIGH);
  digitalWrite(dir_a, HIGH);  
  analogWrite(pwm_b, 100);
  analogWrite(pwm_a, 255);
  delay(time);
}

void reverse_left(int time){
// Reverse Left
  digitalWrite(dir_a, HIGH);
  digitalWrite(dir_b, LOW);
  analogWrite(pwm_b, 100);
  analogWrite(pwm_a, 255); 
  delay(time);
}

void reset(){
  analogWrite(pwm_b, 0);
  analogWrite(pwm_a, 0); 
}

void send_command(int command, int time){
  switch (command){

     //reset command
     case 0: reset(); break;

     // single command
     case 1: forward(time); break;
     case 2: reverse(time); break;
     case 3: right(time); break;
     case 4: left(time); break;

     //combination command
     case 6: forward_right(time); break;
     case 7: forward_left(time); break;
     case 8: reverse_right(time); break;
     case 9: reverse_left(time); break;

     default: Serial.print("Invalid Command\n");
    }
}
