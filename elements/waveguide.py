import torch
from .base import OpticalElement
from core.propagation import OpticalPropagation


class Waveguide(OpticalElement):
    """Simple waveguide element with recurrent reflection.

    Parameters
    ----------
    resolution : tuple
        Spatial resolution (H, W).
    pixel_pitch : float
        Pixel pitch in meters.
    wavelength : float
        Operating wavelength in meters.
    transmission : float
        Fraction of the field that directly exits without entering the
        waveguide. Corresponds to :math:`p_1` in the user description.
    coupling : float
        Fraction of the waveguided field that exits the guide on each
        round trip. This acts like :math:`p_2, p_3, ...` when assumed
        constant.
    reflection : float
        Amplitude reflection coefficient at the distant surface.
    waveguide_length : float
        Physical length of the guide (distance between reflective
        surfaces) in meters. Each round trip propagates this distance
        twice (forth and back).
    max_roundtrips : int
        Maximum number of internal reflections to simulate.
    """

    def __init__(
        self,
        resolution,
        pixel_pitch,
        wavelength,
        transmission=0.5,
        coupling=0.5,
        reflection=0.9,
        waveguide_length=1e-3,
        max_roundtrips=3,
        name=None,
    ):
        self.transmission = transmission
        self.coupling = coupling
        self.reflection = reflection
        self.waveguide_length = waveguide_length
        self.max_roundtrips = max_roundtrips
        super().__init__(resolution, pixel_pitch, wavelength, name)

    def _init_parameters(self):
        self.propagator = OpticalPropagation()

    def forward(self, field):
        field = self.ensure_complex(field)
        output = self.transmission * field
        wg_field = (1 - self.transmission) * field
        for _ in range(self.max_roundtrips):
            if torch.abs(wg_field).max() < 1e-8:
                break
            wg_field = self.propagator.propagate(
                wg_field, self.waveguide_length, self.wavelength, self.pixel_pitch
            )
            wg_field = self.reflection * wg_field
            wg_field = self.propagator.propagate(
                wg_field, self.waveguide_length, self.wavelength, self.pixel_pitch
            )
            leaked = self.coupling * wg_field
            output = output + leaked
            wg_field = (1 - self.coupling) * wg_field
        return output

    def visualize(self):
        return {}
