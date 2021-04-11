def calculate_iou(gt, pr) -> float:
    dx = max(min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1, 0)
    dy = max(min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1, 0)

    overlap_area = dx * dy
    union_area = ((gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) + (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) - overlap_area)

    return overlap_area / union_area


def find_best_match(pred, gts, threshold) -> bool:
    found_match = False
    max_iou = 0
    best_gt_index = -1

    for i, gt in enumerate(gts):
        if gt is not None:
            iou = calculate_iou(pred, gt)
            found_match |= (iou >= threshold)
            if iou > max_iou and found_match:
                best_gt_index = i
                max_iou = iou

    if best_gt_index != -1:
        gts[best_gt_index] = None

    return found_match


def calculate_metric(gts, preds, threshold) -> float:
    tp, fp, fn = 0, 0, 0

    for pred in preds:
        if find_best_match(pred, gts, threshold):
            tp += 1
        else:
            fp += 1

    for gt in gts:
        if gt is not None:
            fn += 1

    return tp / (tp + fp + fn)


def calculate_ious(gts, preds, thresholds) -> list:
    precision_on_threshold = []

    for threshold in thresholds:
        precision_at_threshold = calculate_metric(gts.copy(), preds, threshold)
        precision_on_threshold.append(precision_at_threshold)

    return precision_on_threshold
