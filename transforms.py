from torchvision.transforms import transforms
from config.config import ExperimentConfig

configs = ExperimentConfig().get_config() # 导入配置文件
img_h = configs['Data_Processing']['img_h']  # 从配置文件中获取img_h
img_w = configs['Data_Processing']['img_w']  # 从配置文件中获取img_w

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.Pad(10),

    transforms.RandomCrop((img_h, img_w)),
    # T.Random2DTranslation(height, width),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])
