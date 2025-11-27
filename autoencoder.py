import torch
import torch.nn as nn
import torch.optim as optim
from augmentation import create_augmentations
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from time import time

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, latent_dim)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*7*7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def compute_basic_loss(self, original, reconstructed):
        return nn.MSELoss()(reconstructed, original)
    
    def compute_contrastive_loss(self, original, reconstructed, latent_original, latent_augmented, lambda_val, temperature=0.1):
        recon_loss = nn.MSELoss()(reconstructed, original)
        
        batch_size = latent_original.shape[0]
        
        # Нормализуем векторы
        latent_original_norm = torch.nn.functional.normalize(latent_original, dim=1)
        latent_augmented_norm = torch.nn.functional.normalize(latent_augmented, dim=1)
        
        # Вычисляем попарные сходства
        similarity_matrix = torch.matmul(latent_original_norm, latent_augmented_norm.T) / temperature
        
        # Позитивные пары - диагональ
        positives = torch.diag(similarity_matrix)
        
        # InfoNCE loss
        numerator = torch.exp(positives)
        denominator = torch.sum(torch.exp(similarity_matrix), dim=1)
        contrast_loss = -torch.mean(torch.log(numerator / denominator))
        
        total_loss = recon_loss + lambda_val * contrast_loss
        return total_loss, recon_loss, contrast_loss
    
    def train_model(self, train_loader, lambda_val=0.1, epochs=10, device='cpu', lr=0.001, temperature=0.1):
        """Метод для обучения модели с температурой"""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            start_time = time()
            self.train()
            total_loss = 0
            total_recon_loss = 0
            total_contrast_loss = 0
            
            for batch in train_loader:
                original_batch = batch[0].to(device)
                augmented_batch = create_augmentations(original_batch)
                
                reconstructed, latent_orig = self(original_batch)
                _, latent_aug = self(augmented_batch)
                
                # Передаем temperature в compute_contrastive_loss
                loss, recon_loss, contrast_loss = self.compute_contrastive_loss(
                    original_batch, reconstructed, latent_orig, latent_aug, 
                    lambda_val, temperature=temperature
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_recon_loss += recon_loss.item()
                total_contrast_loss += contrast_loss.item()
                total_loss += loss.item()
            
            finish_time = time()
            epoch_time = finish_time - start_time
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, '
                f'Recognition_Loss: {total_recon_loss/len(train_loader):.4f}, '
                f'Contrast_Loss: {total_contrast_loss/len(train_loader):.4f}, '
                f'Temperature: {temperature}, '
                f'Epoch Time: {epoch_time:.2f}')
        
        return self
        
    def visualize_reconstructions(self, test_images, lambda_val, temperature):
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            original = test_images[:8].to(device)
            augmented = create_augmentations(original)
            reconstructed, _ = self(original)
            
            original = original.cpu()
            augmented = augmented.cpu()
            reconstructed = reconstructed.cpu()
            
            fig, axes = plt.subplots(3, 8, figsize=(16, 6))
            
            for i in range(8):
                axes[0, i].imshow(original[i].squeeze(), cmap='gray')
                axes[0, i].set_title('Оригинал')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(augmented[i].squeeze(), cmap='gray')
                axes[1, i].set_title('Аугментированная')
                axes[1, i].axis('off')
                
                axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
                axes[2, i].set_title('Восстановленная')
                axes[2, i].axis('off')
            
            plt.suptitle(f'Реконструкции (λ={lambda_val}, t={temperature}))')
            plt.tight_layout()
            plt.savefig(f'reconstructions_lambda_{lambda_val}_temp_{temperature}.png')
            #plt.show()
            plt.close()
    
    def visualize_latent_space(self, test_loader, lambda_val, temperature):
        self.eval()
        device = next(self.parameters()).device
        latents = []
        labels = []
        
        with torch.no_grad():
            for batch, label in test_loader:
                batch = batch.to(device)
                _, latent = self(batch)
                latents.append(latent.cpu().numpy())
                labels.append(label.numpy())
        
        latents = np.vstack(latents)
        labels = np.hstack(labels)
        
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latents)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'Латентное пространство (λ={lambda_val}, t={temperature})')
        plt.savefig(f'latent_space_lambda_{lambda_val}_temp_{temperature}.png')
        #plt.show()
        plt.close()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self