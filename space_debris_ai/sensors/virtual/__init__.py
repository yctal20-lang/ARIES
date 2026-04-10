"""
Virtual sensor modules for Space Debris Collector AI System.

These sensors generate realistic noisy readings from simulation ground truth,
providing the data that ML models need for training and inference.
"""

from .gps import VirtualGPS, GPSReading
from .star_tracker import VirtualStarTracker, StarTrackerReading
from .radar import VirtualRadar, RadarReading
from .power import VirtualPowerSensor, PowerReading
from .magnetometer import VirtualMagnetometer, MagnetometerReading
from .sun_sensor import VirtualSunSensor, SunSensorReading
from .proximity import VirtualProximitySensor, ProximityReading
from .thermal import VirtualThermalArray, ThermalReading
from .doppler import VirtualDopplerSensor, DopplerReading
from .radiation import VirtualRadiationSensor, RadiationReading
from .spectrometer import VirtualSpectrometer, SpectrometerReading
from .accelerometer import VirtualAccelerometer, AccelerometerReading
from .hub import VirtualSensorHub

__all__ = [
    "VirtualGPS", "GPSReading",
    "VirtualStarTracker", "StarTrackerReading",
    "VirtualRadar", "RadarReading",
    "VirtualPowerSensor", "PowerReading",
    "VirtualMagnetometer", "MagnetometerReading",
    "VirtualSunSensor", "SunSensorReading",
    "VirtualProximitySensor", "ProximityReading",
    "VirtualThermalArray", "ThermalReading",
    "VirtualDopplerSensor", "DopplerReading",
    "VirtualRadiationSensor", "RadiationReading",
    "VirtualSpectrometer", "SpectrometerReading",
    "VirtualAccelerometer", "AccelerometerReading",
    "VirtualSensorHub",
]
