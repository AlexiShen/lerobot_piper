#!/usr/bin/env python3

import serial
import time
import struct

def send_feetech_command(ser, motor_id, instruction, address=None, data=None):
    """Send a Feetech protocol command"""
    if data is None:
        data = []
    
    # Build packet
    packet = [0xFF, 0xFF, motor_id]
    
    if address is not None:
        length = 3 + len(data)  # ID + INSTRUCTION + ADDRESS + DATA + CHECKSUM
        packet.extend([length, instruction, address])
        packet.extend(data)
    else:
        length = 2 + len(data)  # ID + INSTRUCTION + DATA + CHECKSUM
        packet.extend([length, instruction])
        packet.extend(data)
    
    # Calculate checksum
    checksum = (~sum(packet[2:])) & 0xFF
    packet.append(checksum)
    
    # Send packet
    ser.write(bytes(packet))
    time.sleep(0.01)  # Small delay
    
    return packet

def read_feetech_response(ser, timeout=0.1):
    """Read Feetech protocol response"""
    start_time = time.time()
    response = []
    
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            response.extend(data)
            
            # Check if we have a complete packet
            if len(response) >= 4:
                # Look for header
                for i in range(len(response) - 3):
                    if response[i] == 0xFF and response[i+1] == 0xFF:
                        if i + 4 < len(response):
                            length = response[i+3]
                            if i + 3 + length < len(response):
                                packet = response[i:i+4+length]
                                return packet
        time.sleep(0.001)
    
    return response if response else None

def scan_motors_comprehensive():
    """Comprehensive motor scan with different baud rates and commands"""
    baud_rates = [1000000, 115200, 57600, 38400, 19200, 9600]
    motor_ids = range(0, 254)  # Broadcast and all possible IDs
    
    print("Comprehensive Motor Scan")
    print("=" * 50)
    
    for baud in baud_rates:
        print(f"\nTesting baud rate: {baud}")
        try:
            ser = serial.Serial('/dev/ttyACM0', baud, timeout=0.1)
            time.sleep(0.1)  # Let port stabilize
            
            found_motors = []
            
            # Test ping command for each motor ID
            for motor_id in motor_ids:
                try:
                    # Clear buffer
                    ser.reset_input_buffer()
                    
                    # Send ping (instruction 0x01)
                    packet = send_feetech_command(ser, motor_id, 0x01)
                    
                    # Read response
                    response = read_feetech_response(ser)
                    
                    if response and len(response) >= 6:
                        if response[0] == 0xFF and response[1] == 0xFF and response[2] == motor_id:
                            found_motors.append(motor_id)
                            print(f"  ✓ Motor ID {motor_id} responded: {[hex(b) for b in response]}")
                    
                except Exception as e:
                    continue
            
            if found_motors:
                print(f"  Found motors at baud {baud}: {found_motors}")
                
                # Try to read more info from found motors
                for motor_id in found_motors[:3]:  # Limit to first 3 to avoid spam
                    try:
                        # Read model number (address 0x03)
                        ser.reset_input_buffer()
                        send_feetech_command(ser, motor_id, 0x02, 0x03, [0x02])
                        response = read_feetech_response(ser)
                        if response:
                            print(f"    Motor {motor_id} model response: {[hex(b) for b in response]}")
                            
                        # Read status (address 0x24)
                        ser.reset_input_buffer()
                        send_feetech_command(ser, motor_id, 0x02, 0x24, [0x01])
                        response = read_feetech_response(ser)
                        if response:
                            status = response[5] if len(response) > 5 else 0
                            print(f"    Motor {motor_id} status: 0x{status:02X}")
                            if status != 0:
                                print(f"      Error bits: {bin(status)}")
                    except Exception as e:
                        print(f"    Error reading motor {motor_id} details: {e}")
            else:
                print(f"  No motors found at baud {baud}")
            
            ser.close()
            
        except Exception as e:
            print(f"  Error with baud {baud}: {e}")

if __name__ == "__main__":
    scan_motors_comprehensive()