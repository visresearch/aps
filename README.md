## Asymmetric Patch Sampling for Contrastive Learning

PyTorch implementation and pre-trained models for paper APS: **Asymmetric Patch Sampling for Contrastive Learning**.

<p align="center"><img src="./images/motivation.png" width="50%"  /> </p>

APS is a novel asymmetric patch sampling strategy for contrastive learning, to further boost the appearance asymmetry for better representations. APS significantly outperforms the existing self-supervised methods on both ImageNet-1K and CIFAR dataset, e.g., 2.5% finetune accuracy improvement on CIFAR100. Additionally, compared to other self-supervised methods, APS is more efficient on both memory and computation during training.

[[Paper](https://www.sciencedirect.com/science/article/pii/S0031320324007635)]    [[Arxiv](https://arxiv.org/abs/2306.02854)]    [[BibTex](#Citation)]   [[Model](https://huggingface.co/visresearch/aps/tree/main)]  


### Requirements

---

```
conda create -n asp python=3.9
pip install -r requirements.txt
```

### Datasets

---

Torchvision provides `CIFAR10`, `CIFAR100` datasets. The root paths of data are respectively set to `./dataset/cifar10` and `./dataset/cifar100`. `ImageNet-1K` dataset is placed at `./dataset/ILSVRC`.

### Pre-training

---

To start the APS pre-training, simply run the following commands.

#### •  Arguments

- `arch` is the architecture of the pre-trained models，you can choose `vit-tiny`, `vit-small` and `vit-base`.
- `dataset` is the pre-trained dataset.
- `data-root` is the path of the dataset.
- `nepoch` is the pre-trained epochs.

Run APS with `ViT-Small/2` network on a single node on `CIFAR100` for 1600 epochs with the following command.

```bash
python main_pretrain.py --arch='vit-small' --dataset='cifar100' --data-root='./dataset/cifar100' --nepoch=1600
```

### Finetuning

---

To finetune `ViT-Small/2` on `CIFAR100`  with the following command.

```bash
python main_finetune.py --arch='vit-small' --dataset='cifar100' --data-root='./dataset/cifar100'  \
                   --pretrained-weights='./weight/pretrain/cifar100/small_1600ep_5e-4_100.pth'
```

### Main Results

---
+ **CIFAR10 and  CIFAR100**

|      Dataset      | Training (#Epochs) |                                   ViT-Tiny/2                                   |                                   ViT-Small/2                                   |                                   ViT-Base/2                                   |
| :----------------: | :----------------: | :-----------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
| **CIFAR10** | **Accuracy**(1600) |                                      97.2%                                      |                                      98.1%                                      |                                      98.2%                                      |
|                    | **Accuracy**(3200) |                                      97.5%                                      |                                      98.2%                                      |                                      98.3%                                      |
| **CIFAR100** | **Accuracy** (1600) |                                      83.4%                                      |                                      84.9%                                      |                                      85.9%                                      |
|                    | **Accuracy** (3200) |                                      83.4%                                      |                                      85.3%                                      |                                      86.0%                                      |


### LICENSE

---

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.


### Citation

```bibtex
@article{shen2025asymmetric,
      title={Asymmetric Patch Sampling for Contrastive Learning}, 
      author={Shen, Chengchao and Chen, Jianzhong and Wang, Shu and Kuang, Hulin and Liu, Jin and Wang, Jianxin},
      journal={Pattern Recognition},
      year={2025}
}
```
