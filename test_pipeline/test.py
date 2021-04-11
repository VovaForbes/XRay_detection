import torch
from tqdm import tqdm
import pandas as pd


def test(model, test_loader, result_df_path):
    model.eval()
    with torch.no_grad():
        result_df = pd.DataFrame({"image_id": [], "x_min": [], "y_min": [], "x_max": [],
                                  "y_max": [], "class_id": [], "confidence": []})
        for images, image_names in tqdm(test_loader):
            images = torch.stack(list(images))
            images = images.cuda()
            with torch.cuda.amp.autocast():
                outputs = model(images)

            for i, image in enumerate(images):
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()
                labels = outputs[i]['labels'].data.cpu().numpy()
                for j, box in enumerate(boxes):
                    new_line = pd.DataFrame({"image_id": image_names[i], "x_min": [box[0]], "y_min": [box[1]],
                                             "x_max": [box[2]], "y_max": [box[3]], "class_id": [labels[j]],
                                             "confidence": [scores[j]]})
                    result_df = result_df.append(new_line)
    result_df.to_csv(result_df_path)
