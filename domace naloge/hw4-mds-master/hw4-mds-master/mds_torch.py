import torch
import torch.nn as nn
import yaml

class MDS1D(nn.Module):
    def __init__(self, items):
        super().__init__()
        # Ustvarimo parametre, ki se bodo optimizirali
        # PyTorch poskrbi za naključno inicializacijo in gradient
        self.items = list(items)
        self.pos = nn.Parameter(torch.rand(len(items), 1) * 200 - 100)
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}

    def forward(self, x):
        # Vrne koordinato mesta x
        idx = self.item_to_idx[x]
        return self.pos[idx]

    def loss(self, data):
        total_loss = 0.0
        for (m1, m2), d_real in data.items():
            pos1 = self.forward(m1)
            pos2 = self.forward(m2)
            
            # Razdalja na premici
            d_map = torch.abs(pos1 - pos2)
            
            total_loss += (d_map - d_real)**2
            
        return total_loss / len(data)

class MDS2D(nn.Module):
    def __init__(self, items):
        super().__init__()
        self.items = list(items)
        # 2D koordinate: (len, 2)
        self.pos = nn.Parameter(torch.rand(len(items), 2) * 200 - 100)
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}

    def forward(self, x):
        idx = self.item_to_idx[x]
        return self.pos[idx]

    def loss(self, data):
        total_loss = 0.0
        for (m1, m2), d_real in data.items():
            pos1 = self.forward(m1)
            pos2 = self.forward(m2)
            
            # 2D Evklidska razdalja
            d_map = torch.sqrt(torch.sum((pos1 - pos2)**2))
            
            total_loss += (d_map - d_real)**2
            
        return total_loss / len(data)

# Za MDS2DReg samo deduješ od MDS2D in popraviš loss
class MDS2DReg(MDS2D):
    def loss(self, data):
        # Pokliči loss od MDS2D in dodaj še penalizacijo za y (index 1)
        base_loss = super().loss(data)
        y_penalty = torch.sum(self.pos[:, 1]**2)
        return base_loss + (0.1 * y_penalty)

def train(model, distances, learning_rate=0.1, n_epochs=500):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for k in range(n_epochs):
        optimizer.zero_grad()      # Počisti stare gradiente
        loss = model.loss(distances)
        loss.backward()            # Izračunaj nove gradiente
        optimizer.step()           # Posodobi parametre
        
        if k % 100 == 0:
            print(f"{k:3} Loss: {loss.item():5.3f}")
    return model