import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#import torch_directml  # Для твоей встройки Radeon 780M

# Импортируем твои модули
from model import get_encoder
from metrics import evaluate_knn_and_diversity

def run_only_metrics():
    # 1. Настройка устройства (выбираем 780M через DirectML или CPU)
    device = torch.device('cpu')
    print("💻 Используем процессор для расчета метрик")

    image_size = 32
    weights_path = 'byol_final_model.pt'

    # 2. Инициализация модели (точно такой же ResNet, как при обучении)
    # pretrained=False, так как мы сейчас загрузим свои веса
    resnet = get_encoder('resnet18', pretrained=False)
    
    try:
        resnet.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print(f"✅ Веса успешно загружены из {weights_path}")
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {weights_path} не найден!")
        return

    resnet.to(device)

    # 3. Подготовка данных (чистые картинки без искажений)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Загружаем CIFAR-10 (он уже скачан в папку /data)
    train_eval_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Ставим батч побольше (например, 64), так как градиенты считать не надо
    train_loader = DataLoader(train_eval_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4. Запуск оценки
    print("\nНачинаем расчет метрик (k-NN и Diversity)...")
    evaluate_knn_and_diversity(resnet, train_loader, test_loader, device)

if __name__ == "__main__":
    run_only_metrics()
