#!/usr/bin/env python3

import serial
import time
import sys

def test_serial_port(port="/dev/ttyACM0", baudrate=1000000):
    """Test basic serial communication"""
    try:
        print(f"Testing serial port {port} at {baudrate} baud...")
        ser = serial.Serial(port, baudrate, timeout=1)
        print("✓ Serial port opened successfully")
        
        # Test basic write/read
        ser.write(b'\xFF\xFF\x01\x02\x01\xFB')  # Simple ping command
        time.sleep(0.1)
        
        if ser.in_waiting > 0:
            response = ser.read_all()
            print(f"✓ Received response: {response.hex()}")
        else:
            print("⚠ No response received")
            
        ser.close()
        print("✓ Serial port closed")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_feetech_bus():
    """Test Feetech motor bus communication"""
    try:
        from lerobot.common.motors.feetech import FeetechMotorsBus
        from lerobot.common.motors import Motor, MotorNormMode
        
        print("\nTesting Feetech bus communication...")
        
        # Create a simple bus with one motor for testing
        bus = FeetechMotorsBus(
            port="/dev/ttyACM0",
            motors={
                "test_motor": Motor(1, "sts3215", MotorNormMode.RADIANS)
            }
        )
        
        print("✓ FeetechMotorsBus created")
        
        # Try to connect
        bus.connect()
        print("✓ Bus connected")
        
        # Try to scan for motors
        print("\nScanning for motors...")
        for motor_id in range(1, 8):  # Check IDs 1-7
            try:
                # Try to read model number
                result = bus.read("Model_Number", motor_id)
                if result is not None:
                    print(f"✓ Found motor at ID {motor_id}: Model {result}")
                else:
                    print(f"⚠ No response from motor ID {motor_id}")
            except Exception as e:
                print(f"✗ Error reading motor ID {motor_id}: {e}")
        
        bus.disconnect()
        print("✓ Bus disconnected")
        return True
        
    except Exception as e:
        print(f"✗ Feetech bus error: {e}")
        return False

if __name__ == "__main__":
    print("STS3215 Servo Communication Test")
    print("=" * 40)
    
    # Test 1: Basic serial communication
    serial_ok = test_serial_port()
    
    # Test 2: Feetech bus communication
    if serial_ok:
        feetech_ok = test_feetech_bus()
    else:
        print("Skipping Feetech test due to serial port issues")
    
    print("\n" + "=" * 40)
    print("Test complete!")