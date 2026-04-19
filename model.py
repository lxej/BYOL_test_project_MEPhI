import torchvision.models as models
import torch.nn as nn

def get_encoder(model_name='resnet18', pretrained=False):
    """
    Загружает базовую архитектуру, которая будет извлекать признаки из картинок.
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError("Пока поддерживаем только resnet18 и resnet50")
    
    # В BYOL нам не нужен финальный слой классификации (Linear), 
    # потому что мы не предсказываем классы (собака/кошка), мы предсказываем векторы.
    # byol-pytorch сам умеет подключаться к нужному слою (по умолчанию 'avgpool'),
    # поэтому мы просто возвращаем модель как есть.
    
    return model
