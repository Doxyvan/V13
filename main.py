'''
2.13  Контрастивный автоэнкодер
Цель. Совместить реконструкцию и контрастивное сближение близких примеров.
Описание. Потеря: L = L_recon + λ · L_contrast, где позитивные пары — разные аугментации одного изображения.
Ход работы. - Базовый автоэнкодер на MNIST/Fashion‑MNIST. - Добавить InfoNCE/CLIP‑style компоненту; λ ∈ {0.1, 0.5, 1.0}. - Визуализировать латентные проекции и реконструкции.
Результат. Влияние λ на структуру латента и качество восстановления.
Подсказки. Косинусная близость и температура улучшают тренировку.
'''

from autoencoder import Autoencoder
from torch import device
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    return train_loader, test_dataset

def main():
    u_device = device('cuda' if is_available() else 'cpu')
    print(f"Используется устройство: {u_device}")
    
    train_loader, test_dataset = get_data_loaders()
    lambda_values = [0.1, 0.5, 1.0]
    temperatures = [0.07, 0.1, 0.2] 
    
    for lambda_val in lambda_values:
        for temp in temperatures:
            print(f"Training with lambda = {lambda_val}, temperature = {temp}")
            
            model = Autoencoder(latent_dim=32).to(u_device)
            model.train_model(train_loader, lambda_val=lambda_val, epochs=5, 
                            device=u_device, temperature=temp)
            
            test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
            test_batch = next(iter(test_loader))[0].to(u_device)
            
            model.visualize_latent_space(test_loader, lambda_val, temp)
            model.visualize_reconstructions(test_batch.cpu(), lambda_val, temp)
            
            model.save_model(f'autoencoder_lambda_{lambda_val}_temp_{temp}.pth')

if __name__ == "__main__":
    main()