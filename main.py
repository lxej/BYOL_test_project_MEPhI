import torch
from torch.utils.data import DataLoader
#from torchvision.datasets import FakeData # Заглушка, пока нет своих фото
from tqdm import tqdm
from torchvision import datasets, transforms # Убедись, что transforms импортирован
from metrics import evaluate_knn_and_diversity # Наш новый файл

# Импортируем наши собственные модули
from model import get_encoder
from augmentations import get_byol_transforms
from visualize import plot_loss
from byol_pytorch import BYOL
import torch_directml

def train():
    # 1. Настройки (Гиперпараметры)
    # Вместо device = torch.device('cuda'...)
    device = torch_directml.device()
    epochs = 10
    batch_size = 8
    lr = 3e-4
    image_size = 32

    print(f"Используем устройство: {device}")

    # 2. Подготовка данных
    train_transform = get_byol_transforms(image_size=image_size)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Данные для обучения (с аугментациями)
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Данные для ТЕСТА (чистые)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False)

    # Данные для МЕТРИК (те же train-картинки, но БЕЗ аугментаций)
    train_eval_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size * 4, shuffle=False)
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
    torch.save(resnet.state_dict(), 'byol_final_model.pt')
    plot_loss(loss_history)

    # --- НОВОЕ: Запуск метрик ---
    print("\nЗапуск финальной оценки качества...")
    # Используем train_eval_loader (чистый), чтобы k-NN не офигел от аугментаций
    evaluate_knn_and_diversity(resnet, train_eval_loader, test_loader, device)

if __name__ == "__main__":
    train()
