import torch
import gloria
import pandas as pd
import os
from . import pretraining_dataset
from torch.utils.data import DataLoader

root = "/scratch/ssc10020/IndependentStudy/SLIP/dataset/ISIC/full_data" 
df = pd.read_csv(gloria.constants.ISIC_test_csv)

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
gloria_model = gloria.load_gloria(device=device)

# generate class prompt
# cls_promts = {
#    'Atelectasis': ['minimal residual atelectasis ', 'mild atelectasis' ...]
#    'Cardiomegaly': ['cardiomegaly unchanged', 'cardiac silhouette enlarged' ...] 
# ...
# } 
cls_prompts = gloria.generate_chexpert_class_prompts()

# process input images and class prompts 
processed_txt = gloria_model.process_class_prompts(cls_prompts, device)

imgs_dataset = pretraining_dataset.ISICTestDataset(root, device)
imgs_loader = DataLoader(
            imgs_dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=32,
            num_workers=18,
        )

# zero-shot classification on 1000 images
similarities = gloria.zero_shot_classification(
    gloria_model, imgs_loader, processed_txt)

import pdb; pdb.set_trace()
print(similarities)