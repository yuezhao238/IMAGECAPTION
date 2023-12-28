from torchvision.models import ResNet101_Weights
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import os
import json
from model import *


# 设置GPU信息
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


checkpoint = '../model/ZhaoModel/best_deepfashion.ckpt'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
model = checkpoint['model']
model.to(device)

if __name__ == '__main__':
    from PIL import Image
    vocab_path = '../data/deepfashion/vocab.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    rev_vocab = {v:k for k,v in vocab.items()}
    while True:
        image_path = input('请输入图片路径：')
        if image_path == 'exit':
            break
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print('图片路径错误')
            continue
        # image = transforms.ToTensor()(image).unsqueeze(0).to(device)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            texts = model.generate_by_beamsearch(image, 5, 100)
        for text in texts:
            text = [rev_vocab[idx] for idx in text]
            print(' '.join(text[1:-1]))


