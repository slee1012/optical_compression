# Enhanced system/builder.py with coordinate system and element spacing
import torch
import torch.nn as nn
import numpy as np
from .coordinate_optimizer import CoordinateOptimizer, CoordinateConfig, print_coordinate_config

class SystemBuilder:
    def __init__(self, config, auto_optimize_coordinates=True, use_roi_system=False):
        self.config = config
        self.auto_optimize_coordinates = auto_optimize_coordinates
        self.use_roi_system = use_roi_system  # New option for ROI-based approach
        self.coordinate_config = None  # Will store optimized coordinate config
        
        self.elements = []
        self.element_specs = []  # For ROI system: store element specifications 
        self.element_positions = []
        self.element_spacings = []  # Free space distances after each element
        self.element_apertures = []  # Clear aperture sizes
        self.refractive_indices = []  # Medium indices between elements
    
    def add_phase_mask(self, position=0.0, spacing_after=5e-3, clear_aperture=10e-3, 
                      medium_index=1.0, learnable=True, init_type='random', **kwargs):
        """Add phase mask with positioning and propagation info.
        
        Args:
            position: Z-position along optical axis (meters)
            spacing_after: Free space distance after element (meters) 
            clear_aperture: Clear aperture diameter (meters)
            medium_index: Refractive index of medium after element
        """
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
        self.element_spacings.append(spacing_after)
        self.element_apertures.append(clear_aperture)
        self.refractive_indices.append(medium_index)
        return self
    
    def add_lens(self, focal_length, position=0.0, spacing_after=5e-3, 
                 aperture_diameter=10e-3, medium_index=1.0, lens_type='thin', **kwargs):
        """Add lens element with positioning and propagation info.
        
        Args:
            focal_length: Lens focal length (meters)
            position: Z-position along optical axis (meters)
            spacing_after: Free space distance after element (meters)
            aperture_diameter: Lens clear aperture diameter (meters)
            medium_index: Refractive index of medium after element
            lens_type: 'thin' or 'thick' lens model
        """
        if self.use_roi_system:
            # For ROI system: store element specifications instead of creating elements
            element_spec = {
                'type': 'ThinLens',  # Map both thin and thick to ThinLens for now
                'focal_length': focal_length,
                'aperture_diameter': aperture_diameter,
                'lens_type': lens_type,
                **kwargs
            }
            self.element_specs.append(element_spec)
            # Still add a placeholder to self.elements for compatibility
            self.elements.append(None)
        else:
            # For traditional system: create elements as before
            from elements.lens import create_thin_lens, ThickLens
            
            if lens_type == 'thin':
                element = create_thin_lens(
                    focal_length=focal_length,
                    aperture_diameter=aperture_diameter,
                    resolution=self.config.sensor_resolution,
                    pixel_pitch=self.config.pixel_pitch,
                    wavelength=self.config.wavelength,
                    **kwargs
                )
            elif lens_type == 'thick':
                element = ThickLens(
                    resolution=self.config.sensor_resolution,
                    pixel_pitch=self.config.pixel_pitch,
                    wavelength=self.config.wavelength,
                    focal_length=focal_length,
                    aperture_diameter=aperture_diameter,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown lens_type: {lens_type}")
            
            self.elements.append(element)
        self.element_positions.append(position)
        self.element_spacings.append(spacing_after)
        self.element_apertures.append(aperture_diameter)  # Use actual aperture diameter
        self.refractive_indices.append(medium_index)
        return self
    
    def add_coded_aperture(self, position=0.0, spacing_after=5e-3, clear_aperture=10e-3,
                          medium_index=1.0, aperture_type='binary', learnable=True, **kwargs):
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
        self.element_spacings.append(spacing_after)
        self.element_apertures.append(clear_aperture)
        self.refractive_indices.append(medium_index)
        return self
    
    def add_lens_array(self, position=0.0, spacing_after=5e-3, clear_aperture=10e-3,
                      medium_index=1.0, n_lenses=(8, 8), lens_type='spherical', **kwargs):
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
        self.element_spacings.append(spacing_after)
        self.element_apertures.append(clear_aperture)
        self.refractive_indices.append(medium_index)
        return self

    def add_waveguide(self, position=0.0, spacing_after=5e-3, clear_aperture=10e-3,
                     medium_index=1.0, transmission=0.5, coupling=0.5,
                     reflection=0.9, waveguide_length=1e-3,
                     max_roundtrips=3, **kwargs):
        from elements.waveguide import Waveguide
        element = Waveguide(
            resolution=self.config.sensor_resolution,
            pixel_pitch=self.config.pixel_pitch,
            wavelength=self.config.wavelength,
            transmission=transmission,
            coupling=coupling,
            reflection=reflection,
            waveguide_length=waveguide_length,
            max_roundtrips=max_roundtrips,
            **kwargs
        )
        self.elements.append(element)
        self.element_positions.append(position)
        self.element_spacings.append(spacing_after)
        self.element_apertures.append(clear_aperture)
        self.refractive_indices.append(medium_index)
        return self
    
    def add_coordinate_break(self, position=0.0, tilt_x=0.0, tilt_y=0.0, 
                           decenter_x=0.0, decenter_y=0.0, spacing_after=0.0):
        """Add coordinate break for tilts and decenters.
        
        Args:
            position: Z-position along optical axis (meters)
            tilt_x, tilt_y: Tilts in radians
            decenter_x, decenter_y: Lateral shifts in meters
            spacing_after: Free space distance after break
        """
        # For now, create a dummy identity element that applies coordinate transform
        coordinate_break = CoordinateBreakElement(tilt_x, tilt_y, decenter_x, decenter_y)
        
        self.elements.append(coordinate_break)
        self.element_positions.append(position)
        self.element_spacings.append(spacing_after)
        self.element_apertures.append(float('inf'))  # No aperture limit
        self.refractive_indices.append(1.0)
        return self
    
    def analyze_system_parameters(self):
        """
        Analyze the planned optical system to extract key parameters for optimization.
        
        Returns:
            tuple: (focal_lengths, propagation_distances, aperture_diameters)
        """
        focal_lengths = []
        propagation_distances = []
        aperture_diameters = []
        
        # Extract focal lengths from lens elements
        for element_class, kwargs in self._get_planned_elements():
            if 'focal_length' in kwargs:
                focal_lengths.append(kwargs['focal_length'])
        
        # Calculate all propagation distances
        if self.element_positions:
            # Distance from origin to first element
            if self.element_positions[0] > 0:
                propagation_distances.append(self.element_positions[0])
            
            # Distances between elements
            for i in range(len(self.element_positions) - 1):
                dist = self.element_positions[i+1] - self.element_positions[i]
                if dist > 0:
                    propagation_distances.append(dist)
            
            # Final propagation distance (from last element)
            if self.element_spacings:
                final_spacing = self.element_spacings[-1]
                if final_spacing > 0:
                    propagation_distances.append(final_spacing)
        
        # Extract aperture diameters
        aperture_diameters = [ap for ap in self.element_apertures if ap < float('inf')]
        
        # Add default values if lists are empty
        if not focal_lengths:
            focal_lengths = [1e-3]  # 1mm default focal length
        if not propagation_distances:
            propagation_distances = [1e-3]  # 1mm default distance
        if not aperture_diameters:
            aperture_diameters = [1e-3]  # 1mm default aperture
        
        return focal_lengths, propagation_distances, aperture_diameters
    
    def _get_planned_elements(self):
        """Get planned element types and parameters (for analysis before building)."""
        # This is a simplified version - in practice, you might want to store
        # element creation parameters during add_* methods
        planned = []
        
        # For now, extract from existing elements (already added)
        for elem in self.elements:
            element_type = type(elem).__name__
            kwargs = {}
            if hasattr(elem, 'focal_length'):
                kwargs['focal_length'] = elem.focal_length
            planned.append((element_type, kwargs))
        
        return planned
    
    def optimize_coordinates(self, verbose=True):
        """
        Optimize coordinate system parameters using input-driven approach.
        
        Args:
            verbose: Whether to print optimization results
            
        Returns:
            CoordinateConfig: Optimized coordinate configuration
        """
        # Analyze system parameters
        focal_lengths, propagation_distances, aperture_diameters = self.analyze_system_parameters()
        
        # Create optimizer with new logic
        optimizer = CoordinateOptimizer(
            wavelength=self.config.wavelength,
            memory_limit_3k=True
        )
        
        if verbose:
            print(f"Starting with input resolution: {self.config.sensor_resolution}")
            print(f"Starting with pixel pitch: {self.config.pixel_pitch*1e6:.1f}μm")
        
        # Use new input-driven optimization
        self.coordinate_config = optimizer.optimize_from_input(
            input_resolution=self.config.sensor_resolution,
            input_pixel_pitch=self.config.pixel_pitch,
            propagation_distances=propagation_distances,
            aperture_diameters=aperture_diameters,
            focal_lengths=focal_lengths
        )
        
        if verbose:
            print_coordinate_config(self.coordinate_config)
        
        # Update config with optimized parameters
        self._apply_optimized_coordinates()
        
        return self.coordinate_config
    
    def _apply_optimized_coordinates(self):
        """Apply optimized coordinate parameters to the system config and rebuild elements."""
        if self.coordinate_config is None:
            return
            
        # Store original values for comparison
        old_resolution = self.config.sensor_resolution
        old_pixel_pitch = self.config.pixel_pitch
            
        # Update config with optimized values
        self.config.sensor_resolution = self.coordinate_config.resolution
        self.config.pixel_pitch = self.coordinate_config.pixel_pitch
        
        # Check if parameters changed significantly
        res_changed = (old_resolution != self.coordinate_config.resolution)
        pitch_changed = abs(old_pixel_pitch - self.coordinate_config.pixel_pitch) > 0.1e-6
        
        if res_changed or pitch_changed:
            print(f"Coordinate optimization changed parameters:")
            if res_changed:
                print(f"  Resolution: {old_resolution} -> {self.coordinate_config.resolution}")
            if pitch_changed:
                print(f"  Pixel pitch: {old_pixel_pitch*1e6:.1f}μm -> {self.coordinate_config.pixel_pitch*1e6:.1f}μm")
            
            # CRITICAL: Rebuild all elements with new parameters
            self._rebuild_elements_with_new_coordinates()
        
        # Store additional optimization info in config if possible
        if hasattr(self.config, 'coordinate_optimization'):
            self.config.coordinate_optimization = self.coordinate_config
            
    def _rebuild_elements_with_new_coordinates(self):
        """Rebuild all optical elements with the new coordinate system parameters."""
        print("Rebuilding optical elements with optimized coordinates...")
        
        # Store element creation info
        element_info = []
        for i, element in enumerate(self.elements):
            element_type = type(element).__name__
            info = {
                'type': element_type,
                'position': self.element_positions[i],
                'spacing': self.element_spacings[i],
                'aperture': self.element_apertures[i],
                'refractive_index': self.refractive_indices[i]
            }
            
            # Extract element-specific parameters
            if hasattr(element, 'focal_length'):
                info['focal_length'] = element.focal_length
            if hasattr(element, 'aperture_diameter'):
                info['aperture_diameter'] = element.aperture_diameter
                
            element_info.append(info)
        
        # Clear existing elements
        self.elements.clear()
        
        # Rebuild elements with new coordinates
        for info in element_info:
            if info['type'] in ['ThinLens', 'ThickLens']:
                from elements.lens import create_thin_lens
                element = create_thin_lens(
                    focal_length=info['focal_length'],
                    aperture_diameter=info.get('aperture_diameter', info['aperture']),
                    resolution=self.config.sensor_resolution,
                    pixel_pitch=self.config.pixel_pitch,
                    wavelength=self.config.wavelength
                )
                self.elements.append(element)
            elif info['type'] == 'PhaseMask':
                from elements.phase_mask import PhaseMask
                element = PhaseMask(
                    resolution=self.config.sensor_resolution,
                    pixel_pitch=self.config.pixel_pitch,
                    wavelength=self.config.wavelength,
                    learnable=True
                )
                self.elements.append(element)
            elif info['type'] == 'CodedAperture':
                from elements.coded_aperture import CodedAperture
                element = CodedAperture(
                    resolution=self.config.sensor_resolution,
                    pixel_pitch=self.config.pixel_pitch,
                    wavelength=self.config.wavelength,
                    learnable=True
                )
                self.elements.append(element)
            # Add more element types as needed
            else:
                print(f"Warning: Cannot rebuild element type {info['type']}")
                
        print(f"Rebuilt {len(self.elements)} optical elements with new coordinates")

    def build(self, auto_draw=True, optimize_coordinates=None):
        """
        Build the optical system and optionally create automatic system layout drawing.
        
        Args:
            auto_draw: Whether to automatically save system layout drawing (default: True)
            optimize_coordinates: Whether to optimize coordinate system (None=auto, True/False=force)
        """
        if len(self.elements) == 0:
            raise ValueError("No optical elements added to system")
        
        # Determine if coordinate optimization should be performed
        should_optimize = optimize_coordinates
        if should_optimize is None:
            should_optimize = self.auto_optimize_coordinates
        
        # Perform coordinate optimization if requested (but not for ROI system)
        if should_optimize and not self.use_roi_system:
            print("Optimizing coordinate system for optical simulation...")
            self.optimize_coordinates(verbose=True)
        
        # Choose system type based on configuration
        if self.use_roi_system:
            from simulation.roi_system import ROIBasedImagingSystem
            print("Using ROI-based dynamic field sizing...")
        else:
            from simulation.coherent_system import ImagingSystem
        
        # Sort elements by position
        positions = self.element_positions
        sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i])
        
        sorted_elements = [self.elements[i] for i in sorted_indices]
        sorted_positions = [positions[i] for i in sorted_indices]
        sorted_spacings = [self.element_spacings[i] for i in sorted_indices]
        sorted_apertures = [self.element_apertures[i] for i in sorted_indices]
        sorted_indices_media = [self.refractive_indices[i] for i in sorted_indices]
        
        # Update sorted lists
        self.elements = sorted_elements
        self.element_positions = sorted_positions
        self.element_spacings = sorted_spacings
        self.element_apertures = sorted_apertures
        self.refractive_indices = sorted_indices_media
        
        # Create appropriate system type
        if self.use_roi_system:
            # For ROI system, we need to sort the element specs as well
            sorted_element_specs = [self.element_specs[i] for i in sorted_indices]
            
            system = ROIBasedImagingSystem(
                initial_resolution=self.config.sensor_resolution,
                initial_pixel_pitch=self.config.pixel_pitch,
                wavelength=self.config.wavelength,
                element_specs=sorted_element_specs,
                element_positions=sorted_positions,
                element_spacings=sorted_spacings,
                element_apertures=sorted_apertures
            )
        else:
            system = ImagingSystem(
                config=self.config,
                elements=self.elements,
                element_positions=self.element_positions,
                element_spacings=self.element_spacings,
                element_apertures=self.element_apertures,
                refractive_indices=self.refractive_indices
            )
        
        # Automatically draw and save system configuration
        if auto_draw:
            self._auto_draw_system_layout()
        
        return system
    
    def _auto_draw_system_layout(self):
        """Automatically create and save system layout visualization."""
        try:
            from analysis.visualization import Visualizer
            visualizer = Visualizer()
            
            # Create element names for better visualization
            element_names = []
            if self.use_roi_system:
                # For ROI system, use element_specs instead of elements
                for spec in self.element_specs:
                    if 'focal_length' in spec:
                        f_mm = spec['focal_length'] * 1000
                        element_names.append(f"{spec['type']}(f={f_mm:.1f}mm)")
                    else:
                        element_names.append(spec['type'])
                # Use element_specs for the drawing
                elements_for_drawing = self.element_specs
            else:
                # For traditional system, use actual elements
                for element in self.elements:
                    if hasattr(element, 'focal_length'):
                        f_mm = element.focal_length * 1000
                        element_names.append(f"{element.__class__.__name__}(f={f_mm:.1f}mm)")
                    else:
                        element_names.append(element.__class__.__name__)
                elements_for_drawing = self.elements
            
            # Draw system geometry
            visualizer.draw_optical_system_geometry(
                elements=elements_for_drawing,
                element_positions=self.element_positions,
                element_spacings=self.element_spacings,
                element_names=element_names,
                system_config=self.config,  # Pass config for sensor/image size info
                show=False,  # Don't show, just save
                save_path='auto'  # Use automatic naming
            )
            
        except Exception as e:
            print(f"Warning: Could not create automatic system layout drawing: {e}")

class PresetSystems:
    @staticmethod
    def simple_phase_mask(config):
        return SystemBuilder(config).add_phase_mask(position=0, learnable=True).build()

class CoordinateBreakElement(nn.Module):
    """Coordinate break element for tilts and decenters."""
    
    def __init__(self, tilt_x=0.0, tilt_y=0.0, decenter_x=0.0, decenter_y=0.0):
        super().__init__()
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y  
        self.decenter_x = decenter_x
        self.decenter_y = decenter_y
        
    def forward(self, field):
        """Apply coordinate transformation to field."""
        # For now, just apply lateral shift using roll
        # Full tilt requires more complex phase transformation
        if abs(self.decenter_x) > 1e-12 or abs(self.decenter_y) > 1e-12:
            # Approximate shift with pixel roll (simplified)
            pixel_pitch = 3.45e-6  # Should get from config
            shift_x = int(self.decenter_x / pixel_pitch)
            shift_y = int(self.decenter_y / pixel_pitch)
            
            if shift_x != 0:
                field = torch.roll(field, shift_x, dims=-1)
            if shift_y != 0:
                field = torch.roll(field, shift_y, dims=-2)
        
        # Tilts would add linear phase ramps - placeholder for future
        return field