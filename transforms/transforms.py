import albumentations as A
from albumentations.pytorch.transforms import ToTensor


def get_train_transform():
    return A.Compose([A.RandomBrightness(),
                      A.RandomContrast(limit=0.5, p=0.5),
                      A.Flip(0.5),
                      ToTensor()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensor()
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
