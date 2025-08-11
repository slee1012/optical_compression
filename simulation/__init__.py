"""
Simulation module for optical imaging systems.

Contains system-level simulation functionality:
- System building and configuration
- Coherent imaging simulation
- Incoherent (spectral) imaging simulation
- Sensor modeling
"""

from .builder import SystemBuilder
from .coherent_system import ImagingSystem
from .incoherent_system import (
    IncoherentImagingSystem, 
    create_daylight_system, 
    create_led_system, 
    create_blackbody_system
)
from .sensor_model import SensorModel

__all__ = [
    'SystemBuilder',
    'ImagingSystem', 
    'IncoherentImagingSystem',
    'create_daylight_system',
    'create_led_system', 
    'create_blackbody_system',
    'SensorModel'
]