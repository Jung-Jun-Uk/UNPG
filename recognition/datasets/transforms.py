from torchvision import transforms
from PIL import Image

def base_transform(img_size, mode):
    assert mode in ['train', 'test']
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size,img_size), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])            
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size,img_size), interpolation=Image.BICUBIC),                    
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform