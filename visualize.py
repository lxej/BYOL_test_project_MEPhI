
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_history, save_path="loss_curve.png", window_size=5):
    """
    Рисует график лосса с добавлением скользящего среднего.
    window_size: размер окна сглаживания (чем больше, тем плавнее линия)
    """
    plt.figure(figsize=(12, 6))
    
    # Рисуем оригинальный (шумный) лосс бледным цветом
    plt.plot(loss_history, alpha=0.3, color='blue', label='Original Loss')
    
    # Считаем скользящее среднее
    if len(loss_history) > window_size:
        # Магия усреднения через свертку
        smoothed_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        
        # Рисуем сглаженную линию ярко и четко
        plt.plot(range(window_size - 1, len(loss_history)), smoothed_loss, 
                 color='blue', linewidth=2, label=f'Smoothed Loss (window={window_size})')

    plt.title("Процесс самообучения BYOL: Анализ сходимости", fontsize=14)
    plt.xlabel("Шаги обучения (Итерации)", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(save_path, dpi=150)
    print(f"График со сглаживанием сохранен в {save_path}")
    plt.close()


def save_metrics_report(accuracy, diversity, save_path="metrics_report.png"):
    plt.figure(figsize=(8, 5))
    plt.axis('off') # Убираем оси
    
    # Рисуем красивую рамку
    plt.text(0.5, 0.8, f"BYOL Self-Supervised Evaluation", 
             fontsize=20, ha='center', weight='bold', color='#2c3e50')
    
    # Метрика 1: Точность
    plt.text(0.5, 0.5, f"k-NN Accuracy (k=5): {accuracy*100:.2f}%", 
             fontsize=18, ha='center', color='#2980b9')
    
    # Метрика 2: Разнообразие
    color_div = '#27ae60' if diversity > 0.1 else '#e74c3c'
    plt.text(0.5, 0.3, f"Feature Diversity (Std): {diversity:.4f}", 
             fontsize=18, ha='center', color=color_div)
    
    # Подпись
    plt.text(0.5, 0.1, "Model: ResNet-18 | Dataset: CIFAR-10", 
             fontsize=12, ha='center', color='#7f8c8d')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
