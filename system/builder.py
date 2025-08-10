# Fixed system/builder.py - NO relative imports
import torch
import torch.nn as nn

class SystemBuilder:
    def __init__(self, config):
        self.config = config
        self.elements = []
        self.element_positions = []
    
    def add_phase_mask(self, position=0.0, learnable=True, init_type='random', **kwargs):
        # Import here to avoid circular imports
        from elements.phase_mask import PhaseMask
        element = PhaseMask(
            resolution=self.config.sensor_resolution,
            pixel_pitch=self.config.pixel_pitch,
            wavelength=self.config.wavelength,
            learnable=learnable,
            init_type=init_type,
            **kwargs
        )
        self.elements.append(element)
        self.element_positions.append(position)
        return self
    
    def add_coded_aperture(self, position=0.0, aperture_type='binary', learnable=True, **kwargs):
        from elements.coded_aperture import CodedAperture
        element = CodedAperture(
            resolution=self.config.sensor_resolution,
            pixel_pitch=self.config.pixel_pitch,
            wavelength=self.config.wavelength,
            aperture_type=aperture_type,
            learnable=learnable,
            **kwargs
        )
        self.elements.append(element)
        self.element_positions.append(position)
        return self
    
    def add_lens_array(self, position=0.0, n_lenses=(8, 8), lens_type='spherical', **kwargs):
        from elements.lens_array import LensArray
        element = LensArray(
            resolution=self.config.sensor_resolution,
            pixel_pitch=self.config.pixel_pitch,
            wavelength=self.config.wavelength,
            n_lenses=n_lenses,
            lens_type=lens_type,
            **kwargs
        )
        self.elements.append(element)
        self.element_positions.append(position)
        return self
    
    def build(self):
        if len(self.elements) == 0:
            raise ValueError("No optical elements added to system")
        
        from system.imaging_system import ImagingSystem
        
        positions = self.element_positions
        sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i])
        self.elements = [self.elements[i] for i in sorted_indices]
        self.element_positions = [positions[i] for i in sorted_indices]
        
        system = ImagingSystem(
            config=self.config,
            elements=self.elements,
            element_positions=self.element_positions
        )
        return system

class PresetSystems:
    @staticmethod
    def simple_phase_mask(config):
        return SystemBuilder(config).add_phase_mask(position=0, learnable=True).build()