from torchvision import transforms

def get_byol_transforms(image_size=256):
    """
    Создает пайплайн жестких аугментаций. 
    В BYOL крайне важно сильно искажать картинки, чтобы сеть училась 
    выделять суть (форму объекта), а не запоминать цвета или фон.
    """
    custom_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)), # Случайный кусок картинки
        transforms.RandomHorizontalFlip(), # Отражение
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # Искажение цветов
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2), # Иногда делаем ЧБ
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return custom_transform

# Класс, который выдает сразу ДВЕ версии одной картинки
class MultiViewDataInjector(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        # Возвращаем два по-разному искаженных варианта
        return [self.transforms(sample), self.transforms(sample)]
