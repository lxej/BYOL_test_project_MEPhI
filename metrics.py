import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from visualize import save_metrics_report  # Импортируем функцию отрисовки

def evaluate_knn_and_diversity(encoder, train_loader, test_loader, device):
    """
    Оценивает качество энкодера с помощью k-NN и проверяет на коллапс.
    """
    encoder.eval()
    
    def extract_features(dataloader, desc):
        features, labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(dataloader, desc=desc):
                imgs = imgs.to(device)
                feats = encoder(imgs)
                features.append(feats.cpu().numpy())
                labels.append(lbls.numpy())
        return np.concatenate(features), np.concatenate(labels)

    print("\n[Метрика] Подготовка данных...")
    train_feats, train_labels = extract_features(train_loader, "Признаки Train")
    test_feats, test_labels = extract_features(test_loader, "Признаки Test")

    # --- МЕТРИКА 1: k-NN Accuracy ---
    print("\n[Метрика 1] Обучение k-NN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_feats, train_labels)
    accuracy = knn.score(test_feats, test_labels)
    
    # --- МЕТРИКА 2: Feature Diversity ---
    std_dev = np.std(test_feats, axis=0).mean()

    # --- ВЫВОД В КОНСОЛЬ ---
    print(f"\n" + "="*30)
    print(f"РЕЗУЛЬТАТЫ ОЦЕНКИ:")
    print(f"Точность k-NN: {accuracy * 100:.2f}%")
    print(f"Разнообразие признаков: {std_dev:.4f}")
    print("="*30)

    # --- ВИЗУАЛИЗАЦИЯ (НОВОЕ) ---
    print("Генерация графического отчета...")
    save_metrics_report(accuracy, std_dev, save_path="metrics_report.png")
    
    encoder.train()
    return accuracy, std_dev
