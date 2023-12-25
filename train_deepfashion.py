import os
import json
import random 
from collections import defaultdict, Counter
from PIL import Image
# from matplotlib import pyplot as plt
from argparse import Namespace 
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import wandb
from torchvision.models import ResNet101_Weights
from utils import *
from model import *
from criterion import PackedCrossEntropyLoss
from metrics.calc_metric import metric_logger

# os.environ["WANDB_API_KEY"] = "6b6ae714ca6908898fec2f0198691c5e2a52b7f7"
# os.environ["WANDB_MODE"] = "offline"

wandb.init(project="imagecaption", name="ARCTIC")

dataset = 'deepfashion'
# 设置模型超参数和辅助变量
config = Namespace(
    max_len = 30,
    captions_per_image = 5,
    batch_size = 2,
    image_code_dim = 2048,
    word_dim = 512,
    hidden_size = 512,
    attention_dim = 512,
    num_layers = 1,
    encoder_learning_rate = 0.0001,
    decoder_learning_rate = 0.0005,
    num_epochs = 20,
    grad_clip = 5.0,
    alpha_weight = 1.0,
    evaluate_step = 900, # 每隔多少步在验证集上测试一次
    checkpoint = None, # 如果不为None，则利用该变量路径的模型继续训练
    best_checkpoint = f'../model/ARCTIC/best_{dataset}.ckpt', # 验证集上表现最优的模型的路径
    last_checkpoint = f'../model/ARCTIC/last_{dataset}.ckpt', # 训练完成时的模型的路径
    beam_k = 5
)

wandb.config.update(config)

metriclogger = metric_logger(wandb)

if not os.path.exists('../model/ARCTIC'):
    os.makedirs('../model/ARCTIC')

# 设置GPU信息
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 数据
data_dir = f'../data/{dataset}/'
vocab_path = f'../data/{dataset}/vocab.json'
train_loader, valid_loader, test_loader = mktrainval(data_dir, vocab_path, config.batch_size, workers=0)

# 模型
with open(vocab_path, 'r') as f:
    vocab = json.load(f)

# 随机初始化 或 载入已训练的模型
start_epoch = 0
checkpoint = config.checkpoint
if checkpoint is None:
    model = ARCTIC(config.image_code_dim, vocab, config.word_dim, config.attention_dim, config.hidden_size, config.num_layers)
else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    model = checkpoint['model']

# 优化器
optimizer = get_optimizer(model, config)

# 将模型拷贝至GPU，并开启训练模式
model.to(device)
model.train()

# 损失函数
loss_fn = PackedCrossEntropyLoss().to(device)

# rouge_l, cider, meteor, bleu
best_rougel, best_cider, best_meteor, best_bleu = 0, 0, 0, 0
print("开始训练")
# fw = open('log.txt', 'w')
# from tqdm import tqdm

global_step = 0
for epoch in range(start_epoch, config.num_epochs):
    instance_step = 0
    # for i, (imgs, caps, caplens) in enumerate(train_loader):
    #     optimizer.zero_grad()
    #     # 1. 读取数据至GPU
    #     imgs = imgs.to(device)
    #     caps = caps.to(device)
    #     caplens = caplens.to(device)

    #     # 2. 前馈计算
    #     predictions, alphas, sorted_captions, lengths, sorted_cap_indices = model(imgs, caps, caplens)
    #     # 3. 计算损失
    #     # captions从第2个词开始为targets
    #     loss = loss_fn(predictions, sorted_captions[:, 1:], lengths)
    #     # 重随机注意力正则项，使得模型尽可能全面的利用到每个网格
    #     # 要求所有时刻在同一个网格上的注意力分数的平方和接近1
    #     if config.alpha_weight:
    #         loss += config.alpha_weight * ((1. - alphas.sum(axis=1)) ** 2).mean()

    #     loss.backward()
    #     # 梯度截断
    #     if config.grad_clip > 0:
    #         nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
    #     # 4. 更新参数
    #     optimizer.step()

    #     wandb.log({"loss": loss.cpu()})
        
    #     # if (i+1) % 100 == 0:
    #     #     print('epoch %d, step %d: loss=%.2f' % (epoch, i+1, loss.cpu()))
    #     #     wandb.log({"epoch": epoch, "step": i+1, "loss": loss.cpu()})
    #         # fw.write('epoch %d, step %d: loss=%.2f \n' % (epoch, i+1, loss.cpu()))
    #         # fw.flush()

    #     instance_step += 1
    #     global_step += 1

    state = {
            'epoch': epoch,
            'step': instance_step,
            'model': model,
            'optimizer': optimizer
            }
        # if (i+1) % config.evaluate_step == 0:
    rouge_l, cider, meteor, bleu = evaluate(valid_loader, model, config, metriclogger, global_step)
    # 5. 选择模型
    # if best_res < bleu_score:
    #     best_res = bleu_score
    #     torch.save(state, config.best_checkpoint)
    print(rouge_l, cider, meteor, bleu)
    print(best_rougel, best_cider, best_meteor, best_bleu)
    if np.mean([rouge_l, cider, meteor, bleu]) > np.mean([best_rougel, best_cider, best_meteor, best_bleu]):
        best_rougel, best_cider, best_meteor, best_bleu = rouge_l, cider, meteor, bleu
        torch.save(state, config.best_checkpoint)

    torch.save(state, config.last_checkpoint)
    # fw.write('Validation@epoch, %d, step, %d, BLEU-4=%.2f\n' % (epoch, i+1, bleu_score))
    # fw.flush()
    # print('Validation@epoch, %d, step, %d, BLEU-4=%.2f' % (epoch, i+1, bleu_score))
    # wandb.log({"epoch": epoch, "step": i+1, "BLEU-4": bleu_score})

checkpoint = torch.load(config.best_checkpoint)
model = checkpoint['model']
rouge_l, cider, meteor, bleu = evaluate(test_loader, model, config, metriclogger, global_step)
# print("Evaluate on the test set with the model that has the best performance on the validation set")
# print('Epoch: %d, BLEU-4=%.2f' % (checkpoint['epoch'], bleu_score))
# wandb.log({"BLEU-4": bleu_score})
# fw.write('Epoch: %d, BLEU-4=%.2f' % (checkpoint['epoch'], bleu_score))
# fw.close()

