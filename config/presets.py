from .system_config import SystemConfig


class WearableGlassesConfig(SystemConfig):
    def __init__(self):
        super().__init__(
            sensor_resolution=(128, 128),
            pixel_pitch=3e-6,
            wavelength=550e-9,
            focal_length=10e-3,
            f_number=2.0,
            propagation_distance=5e-3,
            oversample_factor=1,
            target_compression_ratio=20.0,
            subsample_factor=4,
            learning_rate=1e-4,
            batch_size=8
        )


class HighQualityConfig(SystemConfig):
    def __init__(self):
        super().__init__(
            sensor_resolution=(512, 512),
            pixel_pitch=5e-6,
            wavelength=550e-9,
            focal_length=50e-3,
            f_number=1.8,
            propagation_distance=20e-3,
            oversample_factor=2,
            target_compression_ratio=5.0,
            subsample_factor=1
        )