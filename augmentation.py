import torch
from torchvision.transforms.functional import rotate as tv_rotate, affine

def create_augmentations(images):
    """
    Функция для создания аугментированных версий изображений
    """
    device = images.device
    batch_size = images.shape[0]
    
    # Параметры аугментации
    angles = (torch.rand(batch_size, device=device) * 40 - 20)
    brightness_factors = 0.6 + torch.rand(batch_size, 1, 1, 1, device=device) * 0.8
    translates_x = (torch.rand(batch_size, device=device) * 0.3 - 0.15)
    translates_y = (torch.rand(batch_size, device=device) * 0.3 - 0.15)
    scales = 0.7 + torch.rand(batch_size, device=device) * 0.6
    
    augmented_images = []
    for i in range(batch_size):
        img = images[i]
        angle = angles[i].item()
        
        # Поворот
        rotated_img = tv_rotate(img, angle)
        
        # Сдвиг + масштаб
        translate = (translates_x[i].item() * img.shape[2], translates_y[i].item() * img.shape[1])
        scale = scales[i].item()
        
        transformed_img = affine(rotated_img, angle=0, translate=translate, scale=scale, shear=0)
        
        # Добавление шума
        noise = torch.randn_like(transformed_img) * 0.1
        noisy_img = transformed_img + noise
        
        augmented_images.append(noisy_img)
    
    augmented_batch = torch.stack(augmented_images)
    
    # Корректировка яркости
    augmented_batch = augmented_batch * brightness_factors
    augmented_batch = torch.clamp(augmented_batch, 0, 1)
    
    return augmented_batch