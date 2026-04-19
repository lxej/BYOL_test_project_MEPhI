
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
'''
import matplotlib.pyplot as plt
import torch

def plot_loss(loss_history, save_path="loss_curve.png"):
    """
    Рисует график падения ошибки (Loss) в процессе обучения.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='BYOL Loss', color='blue', linewidth=2)
    plt.title("Процесс самообучения (BYOL Training Loss)")
    plt.xlabel("Шаги обучения (Итерации)")
    plt.ylabel("Loss (MSE между Online и Target)")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"График лосса сохранен в {save_path}")
    plt.close()

def show_augmented_pair(original_img_tensor, view1, view2):
    """
    Показывает оригинал и два искаженных вида, которые идут в сети.
    """
    # Денормализация для красивой отрисовки
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Конвертируем тензоры обратно в картинки для matplotlib
    imgs = [original_img_tensor, inv_normalize(view1), inv_normalize(view2)]
    titles = ["Оригинал", "Вид 1 (Для Онлайн-сети)", "Вид 2 (Для Целевой сети)"]
    
    for ax, img, title in zip(axes, imgs, titles):
        # Переводим формат из (C, H, W) в (H, W, C)
        ax.imshow(img.permute(1, 2, 0).clip(0, 1).numpy())
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

'''
