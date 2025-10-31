#!/usr/bin/env python3

import serial
import time
import struct

def send_feetech_command(ser, motor_id, instruction, address=None, data=None):
    """Send a Feetech protocol command"""
    if data is None:
        data = []
    
    packet = [0xFF, 0xFF, motor_id]
    
    if address is not None:
        length = 3 + len(data)
        packet.extend([length, instruction, address])
        packet.extend(data)
    else:
        length = 2 + len(data)
        packet.extend([length, instruction])
        packet.extend(data)
    
    checksum = (~sum(packet[2:])) & 0xFF
    packet.append(checksum)
    
    ser.write(bytes(packet))
    time.sleep(0.01)
    return packet

def read_feetech_response(ser, timeout=0.1):
    """Read Feetech protocol response"""
    start_time = time.time()
    response = []
    
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            response.extend(data)
            
            if len(response) >= 4:
                for i in range(len(response) - 3):
                    if response[i] == 0xFF and response[i+1] == 0xFF:
                        if i + 4 < len(response):
                            length = response[i+3]
                            if i + 3 + length < len(response):
                                packet = response[i:i+4+length]
                                return packet
        time.sleep(0.001)
    
    return response if response else None

def check_motor_voltage_and_status():
    """Check motor voltage and detailed status"""
    print("Motor Voltage and Status Check")
    print("=" * 40)
    
    try:
        ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.1)
        time.sleep(0.1)
        
        for motor_id in range(1, 8):
            print(f"\nMotor ID {motor_id}:")
            
            try:
                # Read present voltage (address 0x2A, 1 byte)
                ser.reset_input_buffer()
                send_feetech_command(ser, motor_id, 0x02, 0x2A, [0x01])
                response = read_feetech_response(ser)
                
                if response and len(response) >= 6:
                    voltage_raw = response[5]
                    voltage = voltage_raw / 10.0  # Convert to volts
                    print(f"  Voltage: {voltage:.1f}V (raw: {voltage_raw})")
                    
                    if voltage < 6.0:
                        print(f"  ⚠️ WARNING: Voltage too low! (min 6.0V)")
                    elif voltage > 12.0:
                        print(f"  ⚠️ WARNING: Voltage too high! (max 12.0V)")
                    else:
                        print(f"  ✓ Voltage OK")
                else:
                    print(f"  ✗ Failed to read voltage")
                
                # Read temperature (address 0x2B, 1 byte)
                ser.reset_input_buffer()
                send_feetech_command(ser, motor_id, 0x02, 0x2B, [0x01])
                response = read_feetech_response(ser)
                
                if response and len(response) >= 6:
                    temp = response[5]
                    print(f"  Temperature: {temp}°C")
                    if temp > 80:
                        print(f"  ⚠️ WARNING: Temperature high!")
                else:
                    print(f"  ✗ Failed to read temperature")
                
                # Read load (address 0x28, 2 bytes)
                ser.reset_input_buffer()
                send_feetech_command(ser, motor_id, 0x02, 0x28, [0x02])
                response = read_feetech_response(ser)
                
                if response and len(response) >= 7:
                    load_raw = (response[6] << 8) | response[5]
                    load = load_raw if load_raw < 1024 else load_raw - 2048
                    print(f"  Load: {load} (raw: {load_raw})")
                
                # Read status again
                ser.reset_input_buffer()
                send_feetech_command(ser, motor_id, 0x02, 0x24, [0x01])
                response = read_feetech_response(ser)
                
                if response and len(response) >= 6:
                    status = response[5]
                    print(f"  Status: 0x{status:02X}")
                    
                    errors = []
                    if status & 0x01: errors.append("Input Voltage Error")
                    if status & 0x02: errors.append("Angle Limit Error")  
                    if status & 0x04: errors.append("Overheating Error")
                    if status & 0x08: errors.append("Range Error")
                    if status & 0x10: errors.append("Checksum Error")
                    if status & 0x20: errors.append("Overload Error")
                    if status & 0x40: errors.append("Instruction Error")
                    
                    if errors:
                        print(f"  🚨 Errors: {', '.join(errors)}")
                    else:
                        print(f"  ✅ No errors")
                        
            except Exception as e:
                print(f"  ✗ Error reading motor {motor_id}: {e}")
        
        ser.close()
        
    except Exception as e:
        print(f"Serial error: {e}")

def try_clear_errors():
    """Try to clear motor errors by resetting them"""
    print("\n" + "=" * 40)
    print("Attempting to clear motor errors...")
    
    try:
        ser = serial.Serial('/dev/ttyACM0', 1000000, timeout=0.1)
        time.sleep(0.1)
        
        for motor_id in range(1, 8):
            try:
                # Try to write 0 to status register to clear errors
                ser.reset_input_buffer()
                send_feetech_command(ser, motor_id, 0x03, 0x24, [0x00])
                response = read_feetech_response(ser)
                
                if response:
                    print(f"  Motor {motor_id}: Sent error clear command")
                else:
                    print(f"  Motor {motor_id}: No response to error clear")
                    
            except Exception as e:
                print(f"  Motor {motor_id}: Error clearing failed - {e}")
        
        ser.close()
        print("Error clear attempt completed")
        
    except Exception as e:
        print(f"Error clear failed: {e}")

if __name__ == "__main__":
    check_motor_voltage_and_status()
    try_clear_errors()
    print("\nWait 2 seconds and check status again...")
    time.sleep(2)
    check_motor_voltage_and_status()