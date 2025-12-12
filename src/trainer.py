import torch
import torch.nn as nn
import torch.optim as optim
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, train_config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.cfg['learning_rate'])

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def train(self):
        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch()
            # Add validation logic here
            print(f"Epoch {epoch+1}/{self.cfg['epochs']} | Loss: {train_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        path = os.path.join(self.cfg['save_dir'], f"checkpoint_ep{epoch}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")