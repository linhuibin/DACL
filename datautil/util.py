# coding=utf-8
import numpy as np
import torch
from itertools import cycle
from math import sqrt
from collections import Counter
import random
from tqdm import tqdm


def Nmax(test_envs, d):
    for i in range(len(test_envs)):
        if d < test_envs[i]:
            return i
    return len(test_envs)


def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains - num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i, j in zip(meta_train, cycle(meta_test)):
        xi, yi = minibatches[i][0], minibatches[i][1]
        xj, yj = minibatches[j][0], minibatches[j][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches_by_domainperm(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()  # domain id

    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches_by_domainperm1(minibatches):
    pairs = []
    perm1 = torch.randperm(len(minibatches)).tolist()
    for i in range(0, len(minibatches), 2):
        j = random.randint(0, len(minibatches) - 1)
        idx = torch.randperm(len(minibatches[j][0]))

        xi, yi = minibatches[j][0][idx], minibatches[j][1][idx]
        xj, yj = minibatches[perm1[i]][0], minibatches[perm1[i]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches(args, minibatches):
    ld = len(minibatches)
    pairs = []
    tdlist = np.arange(ld)  # minibatch 一个domain
    txlist = np.arange(args.batch_size)  # 一个domain中的数据
    for i in range(ld):
        for j in range(args.batch_size):
            (tdi, tdj), (txi, txj) = np.random.choice(tdlist, 2,
                                                      replace=False), np.random.choice(txlist, 2,
                                                                                       replace=True)  # 取不同域的数据
            if j == 0:
                xi, yi, di = torch.unsqueeze(
                    minibatches[tdi][0][txi], dim=0), minibatches[tdi][1][txi], minibatches[tdi][2][txi]
                xj, yj, dj = torch.unsqueeze(
                    minibatches[tdj][0][txj], dim=0), minibatches[tdj][1][txj], minibatches[tdj][2][txj]
            else:
                xi, yi, di = torch.vstack((xi, torch.unsqueeze(minibatches[tdi][0][txi], dim=0))), torch.hstack(
                    (yi, minibatches[tdi][1][txi])), torch.hstack((di, minibatches[tdi][2][txi]))
                xj, yj, dj = torch.vstack((xj, torch.unsqueeze(minibatches[tdj][0][txj], dim=0))), torch.hstack(
                    (yj, minibatches[tdj][1][txj])), torch.hstack((dj, minibatches[tdj][2][txj]))
        pairs.append(((xi, yi, di), (xj, yj, dj)))
        # x: data
    return pairs


def colorful_spectrum_mix(img1, img2, uniform=1, ratio=1.0, ):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, uniform)

    assert img1.shape == img2.shape
    b, c, h, w = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2  # 0
    w_start = w // 2 - w_crop // 2  # 0

    img1_fft = torch.fft.fft2(img1, dim=(-2, -1))
    img2_fft = torch.fft.fft2(img2, dim=(-2, -1))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    # img1_abs = torch.fft.fftshift(img1_abs, dim=(-2, -1))
    # img2_abs = torch.fft.fftshift(img2_abs, dim=(-2, -1))

    img1_abs_ = torch.clone(img1_abs)
    img2_abs_ = torch.clone(img2_abs)
    img1_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[:, :,
                                                                                                h_start:h_start + h_crop,
                                                                                                w_start:w_start + w_crop]
    img2_abs[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[:, :,
                                                                                                h_start:h_start + h_crop,
                                                                                                w_start:w_start + w_crop]

    # img1_abs = torch.fft.ifftshift(img1_abs, dim=(-2, -1))
    # img2_abs = torch.fft.ifftshift(img2_abs, dim=(-2, -1))

    img21 = img1_abs * (torch.exp(1j * img1_pha))
    img12 = img2_abs * (torch.exp(1j * img2_pha))
    img21 = torch.real(torch.fft.ifftn(img21, dim=(-2, -1)))
    img12 = torch.real(torch.fft.ifftn(img12, dim=(-2, -1)))
    # img21 = np.uint8(np.clip(img21, 0, 255))
    # img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def get_features(loaders, model, device):
    model.eval()
    all_features = []
    all_labels = []
    all_domains = []

    with torch.no_grad():
        for i, loader in enumerate(loaders):
            features_list = []
            labels_list = []
            domains_list = []
            for data in tqdm(loader):
                images, labels, domains = data[0], data[1], data[2]
                features = model.encode_image(images.to(device))
                features_list.append(features)
                labels_list.append(labels)
                domains_list.append(domains)

            all_features.append(torch.cat(features_list).cpu().numpy())
            all_labels.append(torch.cat(labels_list).cpu().numpy())
            all_domains.append(torch.cat(domains_list).cpu().numpy())

    return all_features, all_labels, all_domains


def load_embeddings(cache_file):
    """
    Loads the embeddings from a file
    """
    save_dict = torch.load(cache_file)

    train_features, train_labels, train_domains, eval_features, eval_labels, eval_domains = save_dict['train_features'], \
        save_dict['train_labels'], save_dict['train_domains'], save_dict["eval_features"], save_dict["eval_labels"], \
        save_dict["eval_domains"]
    return train_features, train_labels, train_domains, eval_features, eval_labels, eval_domains


def get_domain_text_embs(model, args, text_prompts):
    """
    Gets the text embeddings of the prompts describing the source and target domains.
    If generic is True, source_text_prompts and target_text_prompts are strings instead of
    templates to put the class name in.
    """
    print(text_prompts)  # list of list
    all_texts = []
    for t in text_prompts:
        texts = [[t.format(c)] for c in args.class_names]
        # 提取每个类别的文本embedding 特征
        text_emb = zeroshot_classifier(model, texts).T
        print(texts, "text_emb", text_emb.shape)
        all_texts.append(text_emb)
    text_pairs = torch.cat(all_texts, dim=-1)
    text_pairs = text_pairs.permute(2, 1, 0)
    print("text pairs", text_pairs.shape)

    mask = torch.ones(len(text_pairs), dtype=torch.bool)
    indices = args.test_envs
    mask[indices] = 0
    # 提取源嵌入（Source Embeddings）
    source_embeddings = text_pairs[mask]
    target_embeddings = text_pairs[indices].unsqueeze(0)
    print("embeddings", source_embeddings.shape)  # [num_domain, num_classes, emb_size]
    print("target embeddings", target_embeddings.shape)

    return source_embeddings, target_embeddings


def zeroshot_classifier(model, prompts):
    """ Computes CLIP text embeddings for a list of prompts."""
    #  return (num_domains, num_classes, emb_size)
    model.eval()
    assert type(prompts[0]) == list, "prompts must be a list of lists"
    with torch.no_grad():
        zeroshot_weights = []
        for texts in tqdm(prompts):
            texts = model.tokenize(texts).cuda()  # tokenize
            class_embedding = model.encode_text(texts)  # embed with text encoder
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights.cpu()
