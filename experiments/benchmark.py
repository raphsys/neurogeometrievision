import torch
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from neurogeomvision import IntegratedVisionSystem

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IntegratedVisionSystem(device=device).to(device)
    model.eval()
    
    input_tensor = torch.randn(32, 3, 32, 32).to(device)
    
    # Warmup
    for _ in range(5):
        _ = model(input_tensor)
        
    # Timing
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model(input_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / 50
    print(f"Average Forward Pass ({device}): {avg_time*1000:.2f} ms")
    print(f"Estimated FPS: {1.0/avg_time:.2f}")

if __name__ == "__main__":
    benchmark()
