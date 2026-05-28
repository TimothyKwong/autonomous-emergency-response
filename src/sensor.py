import struct
import serial
import threading

from dataclasses import dataclass, field
from collections import deque
from typing import Optional

from src.logger import DataLogger

@dataclass
class Measurements:
    # If sr is 10Hz, then 10 samples/sec. Change maxlen based on that.
    light_volt: deque = field(default_factory=lambda: deque(maxlen=15))
    loop_time: deque = field(default_factory=lambda: deque(maxlen=15))

class SensorInterface:
    """
    Handles triggering and reading structured sensor data over a serial connection.
    Expects a fixed binary struct format from the microcontroller.
    """

    STRUCT_FMT = "<fffhffff"
    STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

    def __init__(self, ser: Optional[serial.Serial]):
        self.ser = ser
        self.measurements = Measurements()
        self.datalogger = DataLogger(self.measurements)

    def read(self) -> bool:
        # Fallback
        if self.ser is None:
            self.measurements.light_volt.append(0.0)
            self.measurements.loop_time.append(float('inf'))
            return True
        
        try:
            # Read exactly expected number of bytes
            self.ser.reset_input_buffer()
            data = self.ser.read(self.STRUCT_SIZE)

            if len(data) != self.STRUCT_SIZE:
                print(f"Expected {self.STRUCT_SIZE} bytes, got {len(data)}")
                return False

            unpacked = struct.unpack(self.STRUCT_FMT, data)
            # pressure    = unpacked[0]
            # temp        = unpacked[1]
            # rh          = unpacked[2]
            # light_raw   = unpacked[3]
            light_volt  = unpacked[4]
            # distance    = unpacked[5]
            # speed_sound = unpacked[6]
            loop_time   = unpacked[7]

            # Store needed data 
            self.measurements.light_volt.append(light_volt)
            self.measurements.loop_time.append(loop_time)

            return True

        except (ValueError, IndexError) as e:
            print(f"{e}")
            return False

        except (serial.SerialException, OSError) as e:
            print(f"\033[31mSerial read error: {e}\033[0m")
            return False
        
    def save(self) -> bool:
        if self.ser is None:
            return
        self.datalogger.save_csv()
    
    def get_measurements(self, var: str):
        return getattr(self.measurements, var)
    
class SensorThread:
    """
    Runs a SensorInterface in a background thread, continuously reading measurements.
    """
    def __init__(self, sensor_interface):
        self.sensor = sensor_interface
        self.running = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _loop(self):
        while self.running:
            self.sensor.read()
            self.sensor.save()
