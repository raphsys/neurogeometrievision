import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision import IntegratedVisionSystem

class TestSystem(unittest.TestCase):
    def test_full_pipeline_cpu(self):
        model = IntegratedVisionSystem(input_shape=(3, 32, 32), n_classes=5, use_retina=True, device='cpu')
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        
        self.assertEqual(out['final_output'].shape, (2, 5))
        self.assertIn('ventral_features', out)
        self.assertIn('dorsal_features', out)

    def test_pipeline_no_retina(self):
        # Test direct cortex stimulation mode
        model = IntegratedVisionSystem(input_shape=(3, 32, 32), n_classes=10, use_retina=False, device='cpu')
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out['final_output'].shape, (2, 10))
        # Retina outputs should be empty/irrelevant or handled gracefully
        self.assertEqual(out['retina_outputs'], {})

if __name__ == '__main__':
    unittest.main()
