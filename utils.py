import os
import torch
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from datasets.dataset import ImageTextDataset
import glob
import json

vocab_path = '../data/deepfashion/vocab.json'
with open(vocab_path, 'r') as f:
    vocab = json.load(f)
rev_vocab = {v:k for k,v in vocab.items()}


def get_optimizer(model, config):
    return torch.optim.Adam([{"params": filter(lambda p: p.requires_grad, model.encoder.parameters()), 
                              "lr": config.encoder_learning_rate},
                             {"params": filter(lambda p: p.requires_grad, model.decoder.parameters()), 
                              "lr": config.decoder_learning_rate}])
    
def adjust_learning_rate(optimizer, epoch, config):
    """
        每隔lr_update个轮次，学习速率减小至当前十分之一，
        实际上，我们并未使用该函数，这里是为了展示在训练过程中调整学习速率的方法。
    """
    optimizer.param_groups[0]['lr'] = config.encoder_learning_rate * (0.1 ** (epoch // config.lr_update))
    optimizer.param_groups[1]['lr'] = config.decoder_learning_rate * (0.1 ** (epoch // config.lr_update))


def filter_useless_words(sent, filterd_words):
    # 去除句子中不参与BLEU值计算的符号
    return [w for w in sent if w not in filterd_words]

def evaluate(data_loader, model, config, logger=None, global_step=None):
    model.eval()
    # 存储候选文本
    cands = []
    # 存储参考文本
    refs = []
    # 需要过滤的词
    filterd_words = set({model.vocab['<start>'], model.vocab['<end>'], model.vocab['<pad>']})
    cpi = config.captions_per_image
    device = next(model.parameters()).device
    for i, (imgs, caps, caplens) in enumerate(data_loader):
        if i > 1:
            break
        with torch.no_grad():
            # 通过束搜索，生成候选文本
            texts = model.generate_by_beamsearch(imgs.to(device), config.beam_k, config.max_len+2)
            # 候选文本
            cands.extend([filter_useless_words(text, filterd_words) for text in texts])
            # 参考文本
            refs.extend([filter_useless_words(cap, filterd_words) for cap in caps.tolist()])
    # 实际上，每个候选文本对应cpi条参考文本
    multiple_refs = []
    for idx in range(len(refs)):
        multiple_refs.append(refs[(idx//cpi)*cpi : (idx//cpi)*cpi+cpi])
    # 计算BLEU-4值，corpus_bleu函数默认weights权重为(0.25,0.25,0.25,0.25)
    # 即计算1-gram到4-gram的BLEU几何平均值
    if not logger:

        bleu4 = corpus_bleu(multiple_refs, cands, weights=(0.25,0.25,0.25,0.25))
        model.train()
        return bleu4
    else:
        import numpy as np
        # print(multiple_refs, cands)
        # print(multiple_refs[0], cands[0])
        # tt = [rev_vocab[idx] for idx in cands[0]]
        # print(' '.join(tt))
        # ttt = [rev_vocab[idx] for idx in multiple_refs[0][0]]
        # print(' '.join(ttt))

        # print(multiple_refs, cands)
        # print(np.array(multiple_refs).shape, np.array(cands).shape)
        multiple_refs = [[' '.join([rev_vocab[idx] for idx in ref]) for ref in multiple_refs[i]] for i in range(len(multiple_refs))]
        # print(multiple_refs)
        cands = [' '.join([rev_vocab[idx] for idx in cand]) for cand in cands]
        # print(cands)
        rouge_l, cider, meteor, bleu = logger.log(multiple_refs, cands, global_step)
        model.train()
        return rouge_l, cider, meteor, bleu

def mktrainval(data_dir, vocab_path, batch_size, workers=4):
    train_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_set = ImageTextDataset(os.path.join(data_dir, 'train_data.json'), 
                                 vocab_path, 'train',  transform=train_tx)
    valid_set = ImageTextDataset(os.path.join(data_dir, 'test_data.json'), 
                                 vocab_path, 'val', transform=val_tx)
    test_set = ImageTextDataset(os.path.join(data_dir, 'test_data.json'), 
                                 vocab_path, 'test', transform=val_tx)
    # print(len(train_set), len(valid_set), len(test_set))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    # print(len(train_set), len(valid_set), len(test_set))
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    mktrainval('../data/deepfashion', '../data/deepfashion/vocab.json', 32, workers=0)
