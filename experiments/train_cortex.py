import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os

# Ensure neurogeomvision is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from neurogeomvision import IntegratedVisionSystem
from neurogeomvision.utils import CorticalMetrics, save_activation_grid

def main():
    # Config
    BATCH_SIZE = 64
    EPOCHS = 2
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Using CIFAR10 as sample
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = IntegratedVisionSystem(input_shape=(3, 32, 32), n_classes=10, use_retina=True, device=DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    metrics = CorticalMetrics()

    # Loop
    model.train()
    for epoch in range(EPOCHS):
        metrics.reset()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            out_dict = model(data)
            loss = criterion(out_dict['final_output'], target)
            loss.backward()
            optimizer.step()
            
            metrics.update(out_dict['final_output'], target, out_dict)
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        results = metrics.get_results()
        print(f"Epoch {epoch} Results: {results}")

    # Save
    torch.save(model.state_dict(), "trained_cortex.pth")
    print("Model saved to trained_cortex.pth")

    # Visualize last batch
    save_activation_grid(out_dict['retina_outputs']['retina_p_out'], "logs/retina_p.png")

if __name__ == "__main__":
    main()
