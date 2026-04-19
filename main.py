import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData # Заглушка, пока нет своих фото
from tqdm import tqdm
from torchvision import datasets

# Импортируем наши собственные модули
from model import get_encoder
from augmentations import get_byol_transforms
from visualize import plot_loss
from byol_pytorch import BYOL

def train():
    # 1. Настройки (Гиперпараметры)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    batch_size = 8
    lr = 3e-4
    image_size = 32

    print(f"Используем устройство: {device}")

    # 2. Подготовка данных
    # В реальности здесь будет: datasets.ImageFolder(root='path/to/data', transform=...)
    # Но для теста используем FakeData, чтобы код запустился сразу
    transform = get_byol_transforms(image_size=image_size)

    train_dataset = datasets.CIFAR10(
    root='./data',       # Папка, куда скачаются картинки
    train=True, 
    download=True,       # Программа сама скачает их из интернета при первом запуске
    transform=transform  # Твои аугментации из augmentations.py
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    '''
    dataset = FakeData(size=100, image_size=(3, image_size, image_size), transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    '''
    
    # 3. Инициализация модели
    # Берем ResNet-18 из нашего model.py
    resnet = get_encoder('resnet18', pretrained=True)
    
    # Оборачиваем в BYOL
    learner = BYOL(
        resnet,
        image_size = image_size,
        hidden_layer = 'avgpool',
        projection_size = 256,
        moving_average_decay = 0.99
    ).to(device)

    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)

    # 4. Цикл обучения
    loss_history = []
    
    print("Старт обучения...")
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_losses = []
        
        for images, _ in pbar:
            images = images.to(device)
            
            # Считаем Loss: MSE(предсказание_online, выход_target)
            loss = learner(images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Обновляем Target Network (EMA шаг)
            learner.update_moving_average()
            
            # Сохраняем статистику
            current_loss = loss.item()
            epoch_losses.append(current_loss)
            loss_history.append(current_loss)
            
            pbar.set_postfix(loss=f"{current_loss:.4f}")

    # 5. Финализация
    print("Обучение завершено!")
    
    # Сохраняем веса только базового энкодера (ResNet)
    # Это то, что мы потом будем использовать для классификации
    torch.save(resnet.state_dict(), 'byol_final_model.pt')
    
    # Рисуем график лосса через наш визуализатор
    plot_loss(loss_history)

if __name__ == "__main__":
    train()
