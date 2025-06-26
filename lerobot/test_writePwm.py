import sys
import os
import time

sys.path.append("..")
import scservo_sdk as scs
from scservo_sdk.port_handler import PortHandler

# Default settings
STS_ID = 1  # Change to your servo's ID
DEVICENAME = '/dev/ttyACM0'  # Update as needed
BAUDRATE = 1000000

portHandler = PortHandler(DEVICENAME)
packetHandler = scs(portHandler)

if not portHandler.openPort():
    print("Failed to open port")
    sys.exit(1)
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    sys.exit(1)

# Set to wheel mode (PWM mode)
STS_MODE = 33
packetHandler.write1ByteTxRx(STS_ID, STS_MODE, 1)

# Enable torque
STS_TORQUE_ENABLE = 40
packetHandler.write1ByteTxRx(STS_ID, STS_TORQUE_ENABLE, 1)

print("Press Ctrl+C to quit.")

try:
    while True:
        # Command PWM (speed), e.g., 100 for low torque
        STS_GOAL_SPEED_L = 46
        pwm_value = 100  # Try positive and negative values
        packetHandler.write2ByteTxRx(STS_ID, STS_GOAL_SPEED_L, pwm_value)

        # Read position
        pos, comm_result, comm_error = packetHandler.ReadPos(STS_ID)
        if comm_result == 0 and comm_error == 0:
            print(f"Current Position: {pos}")
            # Read current if needed
            # current, comm_result, comm_error = packetHandler.ReadCurrent(STS_ID)
            # print(f"Current: {current}")
        else:
            print(f"ReadPos error: comm_result={comm_result}, comm_error={comm_error}")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")

portHandler.closePort()