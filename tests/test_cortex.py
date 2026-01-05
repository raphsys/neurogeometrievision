import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision.cortex.areas import V1Area, V2Area, V4Area, MTArea
from neurogeomvision.cortex.pathways import VentralStream, DorsalStream

class TestCortex(unittest.TestCase):
    def setUp(self):
        self.input_tensor = torch.randn(2, 6, 32, 32) # 6 channels from retina

    def test_v1_processing(self):
        v1 = V1Area(in_channels=6, out_channels=16)
        out = v1(self.input_tensor)
        # Complex cells pool stride 2 -> 32/2 = 16
        self.assertEqual(out['combined_response'].shape, (2, 16, 16, 16))

    def test_ventral_stream(self):
        ventral = VentralStream(in_channels=6)
        out = ventral(self.input_tensor)
        # V1(32->16) -> V2(16->16) -> V4(16->8)
        self.assertEqual(out.shape[2:], (8, 8))

    def test_dorsal_stream(self):
        dorsal = DorsalStream(in_channels=6)
        out = dorsal(self.input_tensor)
        # MT has adaptive pool to 4x4
        self.assertEqual(out.shape[2:], (4, 4))

if __name__ == '__main__':
    unittest.main()
