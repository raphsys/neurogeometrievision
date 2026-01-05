import torch
from color import DoubleOpponentCell, ColorOpponency

print("Test DoubleOpponentCell...")
try:
    # Créer une entrée simulée
    color_model = ColorOpponency()
    test_rgb = torch.randn(3, 64, 64)
    opponent = color_model(test_rgb)
    
    # Tester différentes cellules
    for color in ['rg', 'by']:
        for center in ['on', 'off']:
            cell = DoubleOpponentCell(preferred_color=color, center_color=center)
            response = cell(opponent)
            print(f"  {color}_{center}: {response.shape}")
    
    print("✓ DoubleOpponentCell fonctionne!")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
