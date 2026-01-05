import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision.cortex.plasticity import STDPPlasticity, HebbianPlasticity

class TestPlasticity(unittest.TestCase):
    def setUp(self):
        self.weights = torch.rand(10, 5) * 0.1 # [Out, In]
        self.pre = torch.rand(1, 5)            # [B, In]
        self.post = torch.rand(1, 10)          # [B, Out]

    def test_stdp_update(self):
        stdp = STDPPlasticity(lr=0.1)
        new_weights = stdp.update_weights(self.weights.clone(), self.pre, self.post)
        
        # Check shapes match
        self.assertEqual(new_weights.shape, self.weights.shape)
        # Check weights changed
        self.assertFalse(torch.equal(new_weights, self.weights))
        # Check bounds (clamped to 0)
        self.assertTrue(torch.all(new_weights >= 0))

    def test_hebbian_oja(self):
        hebb = HebbianPlasticity(learning_rate=0.01)
        new_weights = hebb.update_weights(self.weights.clone(), self.pre, self.post)
        self.assertEqual(new_weights.shape, self.weights.shape)
        self.assertFalse(torch.equal(new_weights, self.weights))

if __name__ == '__main__':
    unittest.main()
