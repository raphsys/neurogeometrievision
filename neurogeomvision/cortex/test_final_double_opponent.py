import torch
import torch.nn.functional as F
from color import DoubleOpponentCell, ColorOpponency

print("Test DoubleOpponentCell avec correction...")
try:
    # Créer une entrée
    color_model = ColorOpponency()
    test_rgb = torch.randn(3, 64, 64)
    opponent = color_model(test_rgb)
    
    print("Dimensions d'entrée:")
    for key, val in opponent.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: {val.shape}")
    
    # Tester une cellule
    print("\nTest cellule RG_ON:")
    cell = DoubleOpponentCell(preferred_color='rg', center_color='on')
    
    # Vérifier les tailles des filtres
    print(f"  Center filter size: {cell.center_filter.shape}")
    print(f"  Surround filter size: {cell.surround_filter.shape}")
    print(f"  Orientation filter size: {cell.orientation_filter.shape}")
    
    response = cell(opponent)
    print(f"  Response shape: {response.shape}")
    
    # Vérifier que c'est la bonne taille (devrait être proche de 64x64)
    if len(response.shape) >= 2:
        h, w = response.shape[-2], response.shape[-1]
        print(f"  Output size: {h}x{w}")
        if abs(h - 64) <= 4 and abs(w - 64) <= 4:
            print("  ✓ Taille correcte!")
        else:
            print(f"  ⚠️  Taille inhabituelle, attendu ~64x64")
    
    print("\nTest toutes les combinaisons:")
    for color in ['rg', 'by']:
        for center in ['on', 'off']:
            try:
                cell = DoubleOpponentCell(preferred_color=color, center_color=center)
                resp = cell(opponent)
                if len(resp.shape) >= 2:
                    print(f"  {color}_{center}: {resp.shape[-2]}x{resp.shape[-1]}")
                else:
                    print(f"  {color}_{center}: shape {resp.shape}")
            except Exception as e:
                print(f"  {color}_{center}: ERREUR - {e}")
    
    print("\n✓ Test DoubleOpponentCell terminé!")
    
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
