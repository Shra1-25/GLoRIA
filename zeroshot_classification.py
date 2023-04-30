import torch
from torchmetrics import AUROC 
import gloria
import pandas as pd
import os
from gloria.datasets import pretraining_dataset
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import recall_score


root = "/scratch/ssc10020/IndependentStudy/SLIP/dataset/ISIC" 
df = pd.read_csv(gloria.constants.ISIC_test_csv)

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
gloria_model = gloria.load_gloria(name='gloria_resnet50_ISIC_seed_123', device=device)

# generate class prompt
# cls_promts = {
#    'Atelectasis': ['minimal residual atelectasis ', 'mild atelectasis' ...]
#    'Cardiomegaly': ['cardiomegaly unchanged', 'cardiac silhouette enlarged' ...] 
# ...
# } 
cls_prompts = gloria.generate_path_class_prompts()

# process input images and class prompts 
processed_txt = gloria_model.process_class_prompts(cls_prompts, device)

imgs_dataset = pretraining_dataset.ISICTestDataset(root, device, gloria_model.process_img)
imgs_loader = DataLoader(
            imgs_dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=32,
            num_workers=18,
            collate_fn=imgs_dataset.test_data_collate,
        )

# zero-shot classification on 1000 images
similarities, targets = gloria.zero_shot_classification(
    gloria_model, imgs_loader, processed_txt)

# import pdb; pdb.set_trace()
print(similarities)

def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = torch.tensor(output)
    target = torch.tensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

diagnosis_map = gloria.constants.DIAGNOSIS_MAP
class_pred = [diagnosis_map[k] for k in similarities.idxmax(axis='columns')]
rec_score = recall_score(targets, class_pred, labels=[0,1,2,3,4,5,6,7], average='micro')
preds = similarities.to_numpy()
acc_score = accuracy(preds, targets, topk=(1,5))
preds = torch.tensor(preds)
targets = torch.tensor(targets)
torch.save(preds, 'preds_seed_123.pt')
torch.save(targets, 'targets_seed_123.pt')
# import pdb; pdb.set_trace()
# auc_score = AUROC(task='multiclass', num_classes=8)(preds, targets)

print('Accuracy:', acc_score, 'Recall score:', rec_score)