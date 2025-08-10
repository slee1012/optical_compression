import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from elements.waveguide import Waveguide


def test_waveguide_pass_through():
    wg = Waveguide((8, 8), 1e-6, 500e-9, transmission=1.0, coupling=0.0,
                   reflection=0.0, max_roundtrips=0)
    field = torch.ones(1, 8, 8, dtype=torch.complex64)
    out = wg(field)
    assert torch.allclose(out, field)


def test_waveguide_single_roundtrip():
    wg = Waveguide((8, 8), 1e-6, 500e-9, transmission=0.0, coupling=1.0,
                   reflection=1.0, waveguide_length=0.0, max_roundtrips=1)
    field = torch.ones(1, 8, 8, dtype=torch.complex64)
    out = wg(field)
    assert torch.allclose(out, field)
