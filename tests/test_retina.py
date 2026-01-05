import unittest
import torch
import sys
import os

# Ajout du dossier parent au path pour importer neurogeomvision
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurogeomvision.retina import BioInspiredRetina, PhotoreceptorLayer, GanglionCellLayer

class TestRetina(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.h, self.w = 32, 32
        self.input_tensor = torch.randn(self.batch_size, self.channels, self.h, self.w)

    def test_photoreceptors_shape(self):
        layer = PhotoreceptorLayer(input_channels=3)
        out = layer(self.input_tensor)
        self.assertIn('combined_response', out)
        self.assertEqual(out['combined_response'].shape, (self.batch_size, 3, self.h, self.w))

    def test_retina_forward_shape(self):
        retina = BioInspiredRetina(input_shape=(3, 32, 32), cortical_shape=(32, 32))
        out = retina(self.input_tensor)
        
        # Vérification des clés de sortie
        self.assertIn('retina_p_out', out)
        self.assertIn('retina_m_out', out)
        self.assertIn('raw_spikes', out)
        
        # Vérification des dimensions
        # P pathway préserve souvent la dimension spatiale (dépend du mapping)
        self.assertEqual(out['retina_p_out'].shape[0], self.batch_size)
        self.assertEqual(out['retina_p_out'].shape[2:], (32, 32))

    def test_ganglion_spikes(self):
        # Test du mode train vs eval pour les spikes
        layer = GanglionCellLayer(in_channels=6)
        dummy_input = torch.randn(self.batch_size, 6, 16, 16)
        
        # Train mode (rates)
        layer.train()
        out_train = layer(dummy_input)
        # Rates are probabilities [0, 1]
        self.assertTrue(torch.all(out_train['p_spikes'] >= 0))
        self.assertTrue(torch.all(out_train['p_spikes'] <= 1))
        
        # Eval mode (binary spikes)
        layer.eval()
        out_eval = layer(dummy_input)
        unique_vals = torch.unique(out_eval['p_spikes'])
        for v in unique_vals:
            self.assertIn(v.item(), [0.0, 1.0])

if __name__ == '__main__':
    unittest.main()
