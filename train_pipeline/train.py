import torch
from tqdm import tqdm
import numpy as np
from plots.plots import Visualizations
from metrics.metrics import calculate_ious


def train(model, num_epochs, train_loader, val_loader, lr, push_visualization, print_stats, use_cuda, thresholds,
          fold, iou_save_threshold, seed):
    if push_visualization:
        vis_train = Visualizations("train", thresholds)
        vis_val = Visualizations("val", thresholds)

    itr_train, itr_val = 0, 0
    best_iou = 0
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                           patience=4, min_lr=1e-9, eps=1e-08)
    scaler = torch.cuda.amp.GradScaler()

    print("Start training on fold {0}".format(fold))

    for epoch in range(num_epochs):
        model.train()
        loss_values = []
        for images, targets in tqdm(train_loader):
            itr_train += 1
            optimizer.zero_grad()
            images = torch.stack(list(images))
            if use_cuda:
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
                images = images.cuda()

            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

            scaler.scale(losses).backward()
            loss_values.append(loss_value)

            if itr_train % 50 == 0:
                if push_visualization:
                    vis_train.plot_loss(np.mean(loss_values), itr_train)
                if print_stats:
                    print("Train loss: {0:.4f}".format(np.mean(loss_values)))
                loss_values.clear()

            scaler.step(optimizer)
            scaler.update()

        model.eval()
        with torch.no_grad():
            val_scores = np.zeros(len(thresholds))
            num_images = 0
            for images, targets in tqdm(val_loader):
                itr_val += 1
                images = torch.stack(list(images))
                if use_cuda:
                    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
                    images = images.cuda()
                with torch.cuda.amp.autocast():
                    outputs = model(images)

                for i, image in enumerate(images):
                    boxes = outputs[i]['boxes'].data.cpu().numpy()
                    scores = outputs[i]['scores'].data.cpu().numpy()
                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    preds_sorted_idx = np.argsort(scores)[::-1]
                    preds_sorted = boxes[preds_sorted_idx]
                    val_scores += calculate_ious(preds_sorted, gt_boxes, thresholds=thresholds)
                    num_images += 1

            val_scores /= num_images
            valid_mean_iou_score = np.mean(val_scores)
            if push_visualization:
                for idx, val_score in enumerate(val_scores):
                    vis_val.plot_iou(val_score, itr_val, idx)
                vis_val.plot_mean_iou(valid_mean_iou_score, itr_val)
            if print_stats:
                for idx, val_score in enumerate(val_scores):
                    print("Validation precision over {0:.2f} threshold: {1:.4f}".format(thresholds[idx], val_score))
                print("Validation precision over all IOU's: {0:.4f}. Fold: {1}".format(valid_mean_iou_score, fold))
            scheduler.step(valid_mean_iou_score, epoch)
            if valid_mean_iou_score > best_iou:
                best_iou = valid_mean_iou_score
                if best_iou > iou_save_threshold:
                    torch.save(model, "Faster_rcnn_precision_{0:.4f}_fold_{1}_seed_{2}".format(valid_mean_iou_score,
                                                                                               fold, seed))
