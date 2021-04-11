import argparse
import pandas as pd
from torch.backends import cudnn
import torch
import torchvision
from train_pipeline.train import train
import numpy as np
from data_loader.data_loader import get_train_val_loader, get_test_loader
from utils.utils import train_val_split_folds
from test_pipeline.test import test


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    iou_thresholds = [x for x in np.arange(0.25, 0.95, 0.1)]
    if args.mode == 'train':
        dataframe = pd.read_csv(args.input_mark)
        train_val_folds = train_val_split_folds(dataframe, args.seed, args.num_folds)
        for i, (df_train, df_val) in enumerate(train_val_folds):
            if args.load_checkpoint is None:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            else:
                model = torch.load(args.load_checkpoint)
            if args.use_cuda:
                model = model.cuda()
            train_loader, val_loader = get_train_val_loader(df_train, df_val, args)
            train(model, args.num_epochs, train_loader, val_loader,
                  args.learning_rate, args.push_visualizations, args.print_stats, args.use_cuda, iou_thresholds,
                  i, args.iou_save_threshold, args.seed)
            if not args.use_k_fold:
                break
        assert list(dataframe.columns) == ['image_id', 'Label', 'x_min', 'y_min', 'x_max', 'y_max'], \
            "Please, check the columns in the input dataframe"
    elif args.mode == 'test':
        assert args.load_checkpoint is not None
        assert args.test_image_dir is not None
        assert args.test_output_df_path is not None
        model = torch.load(args.load_checkpoint)
        if args.use_cuda:
            model = model.cuda()
        test_loader = get_test_loader(args)
        test(model, test_loader, args.test_output_df_path)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--input_dir', type=str, help='Path to directory with images.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--mode', type=str, default="train", help='Working mode. Could be train or test.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for k-fold cross validation. '
                                                                 'Used only in train mode.')
    parser.add_argument('--input_mark', type=str, help='Path to csv file with bounding boxes. Used in train mode only.')
    parser.add_argument('--seed', type=int, help="Seed used for random.")
    parser.add_argument('--num_epochs', type=int, help="Amount of epochs to train.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for optimizer.")
    parser.add_argument('--use_cuda', action='store_true', help="GPU usage enabled.")
    parser.add_argument('--print_stats', action='store_true', help="Print training and validation stats.")
    parser.add_argument('--push_visualizations', action='store_true', help="Push visualizations to visdom server. "
                                                                           "The server should be up independently.")
    parser.add_argument('--use_k_fold', action='store_true', help="Use k-fold cross val or learn model on first fold.")
    parser.add_argument('--save_checkpoints', action='store_true', help="Save checkpoints during training.")
    parser.add_argument('--num_workers', type=int, help="Num workers passed to Dataloader constructor.")
    parser.add_argument('--use_augmentations', action='store_true', help="Use augmentations while training.")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory to put saved checkpoints.")
    parser.add_argument('--iou_save_threshold', type=float, default=0.1, help="Threshold to start saving models.")
    parser.add_argument('--load_checkpoint', type=str, help="Continue learning from specific checkpoint")
    parser.add_argument('--test_output_df_path', type=str, help="Define answers csv-file path.")
    parser.add_argument('--test_image_dir', type=str, help="Folder containing test images.")

    args = parser.parse_args()
    main(args)
