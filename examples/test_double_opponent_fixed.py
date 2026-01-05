import torch
from color import DoubleOpponentCell, ColorOpponency

print("Test DoubleOpponentCell corrigé...")
try:
    # Créer une entrée simulée
    color_model = ColorOpponency()
    test_rgb = torch.randn(3, 64, 64)
    opponent = color_model(test_rgb)
    
    print(f"Input shapes:")
    print(f"  RG_ON: {opponent['rg_on'].shape}")
    print(f"  Luminance: {opponent['luminance'].shape}")
    
    # Tester une cellule
    cell = DoubleOpponentCell(preferred_color='rg', center_color='on')
    response = cell(opponent)
    
    print(f"\n✓ DoubleOpponentCell fonctionne!")
    print(f"  Response shape: {response.shape}")
    
    # Vérifier la taille
    if len(response.shape) == 2:
        print(f"  Output: {response.shape[0]}x{response.shape[1]}")
    elif len(response.shape) == 4:
        print(f"  Output: {response.shape[2]}x{response.shape[3]}")
    
    # Tester toutes les combinaisons
    print(f"\nTest toutes les combinaisons:")
    for color in ['rg', 'by']:
        for center in ['on', 'off']:
            cell = DoubleOpponentCell(preferred_color=color, center_color=center)
            response = cell(opponent)
            if len(response.shape) == 2:
                print(f"  {color}_{center}: {response.shape[0]}x{response.shape[1]}")
            else:
                print(f"  {color}_{center}: shape {response.shape}")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
