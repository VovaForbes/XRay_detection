import torch
from torch.utils.data import Dataset, DataLoader
from transforms.transforms import get_train_transform, get_valid_transform
import cv2
import os


def collate_fn(batch):
    return list(zip(*batch))


class XRayTestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.all_imgs = os.listdir(image_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.image_dir, self.all_imgs[idx])
        image = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = torch.Tensor(image).permute(2, 0, 1) / 255.
        return image, self.all_imgs[idx]


class XRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.class_to_label = dict()
        for i, cls in enumerate(dataframe["Label"].unique()):
            self.class_to_label[cls] = i
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(os.path.join(self.image_dir, image_id), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values.astype(float)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = records.Label.values
        labels = [self.class_to_label[item] for item in labels]

        target = dict()
        target['boxes'] = boxes
        target['labels'] = torch.Tensor(labels).long()

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        else:
            image = torch.Tensor(image).permute(2, 0, 1)
            image /= 255.

        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def get_train_val_loader(df_train, df_val, args):
    if not args.use_augmentations:
        train_dataset = XRayDataset(df_train, args.input_dir)
        val_dataset = XRayDataset(df_val, args.input_dir)
    else:
        train_dataset = XRayDataset(df_train, args.input_dir, get_train_transform())
        val_dataset = XRayDataset(df_val, args.input_dir, get_valid_transform())

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    return train_loader, val_loader


def get_test_loader(args):
    test_dataset = XRayTestDataset(args.test_image_dir)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)
    return test_loader
