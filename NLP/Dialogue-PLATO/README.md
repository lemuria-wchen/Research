# PLATO
**PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable**
[paper link](https://www.aclweb.org/anthology/2020.acl-main.9.pdf)

**\*\*\*\*\* Update \*\*\*\*\***

Jul. 9, 2020: We are opening [PLATO-2](https://github.com/PaddlePaddle/Knover/tree/master/plato-2), a large-scale generative model with latent space for open-domain dialogue systems.

Nov. 14, 2019: Support new APIs in paddlepaddle 1.6 (model files in the link have been updated accordingly), multi-GPU training and decoding strategy of top-k sampling. Release our baseline model `PLATO w/o latent`.

## Requirements
```
- python >= 3.6
- paddlepaddle == 1.6.1
- numpy
- nltk
- tqdm
- visualdl >= 1.3.0 (optional)
- regex
```
Recommend you install to python packages by command: `pip install -r requirement.txt`

## Pre-trained dialogue generation model
A novel pre-training model for dialogue generation is introduced in this work, incorporated with latent discrete variables for one-to-many relationship modeling. Our model is flexible enough to support various kinds of conversations, including chit-chat, knowledge grounded dialogues, and conversational question answering. The pre-training is carried out with Reddit and Twitter corpora. You can download the uncased pre-trained model from:
* PLATO, uncased [model](https://baidu-nlp.bj.bcebos.com/PLATO/model.tar.gz): 12-layers, 768-hidden, 12-heads, 132M parameters
* PLATO w/o latent, uncased [model](https://baidu-nlp.bj.bcebos.com/PLATO/model-baseline.tar.gz): 12-layers 768-hidden, 12-heads, 109M parameters

```bash
mv /path/to/model.tar.gz .
tar xzf model.tar.gz
```

## Fine-tuning
We also provide instructions to fine-tune PLATO on different conversation datasets (chit-chat, knowledge grounded dialogues and conversational question answering).

### Data preparation
Download data from the [link](https://baidu-nlp.bj.bcebos.com/PLATO/data.tar.gz).
The tar file contains three processed datasets: `DailyDialog`, `PersonaChat` and `DSTC7_AVSD`.
```bash
mv /path/to/data.tar.gz .
tar xzf data.tar.gz
```

### Data format
Our model supports two kinds of data formats for dialogue context: `multi` and `multi_knowledge`.
* `multi`: multi-turn dialogue context.
```txt
u_1 __eou__ u_2 __eou__ ... u_n \t r
```
* `multi_knowledge`: multi-turn dialogue context with background knowledges.
```txt
k_1 __eou__ k_2 __eou__ ... k_m \t u_1 __eou__ u_2 __eou__ ... u_n \t r
```

If you want to use this model on other datasets, you can process your data accordingly.

### Train
Fine-tuning the pre-trained model on different `${DATASET}`.
```bash
# DailyDialog / PersonaChat / DSTC7_AVSD
DATASET=DailyDialog
sh scripts/${DATASET}/train.sh
```
After training, you can find the output folder `outputs/${DATASET}` (by default). It contatins `best.model` (best results on validation dataset), `hparams.json` (hyper-parameters of training script) and `trainer.log` (training log).


Fine-tuning the pre-trained model on multiple GPUs.

Note: You need to install NCCL library and set up the environment variable `LD_LIBRARY` properly.
```bash
sh scripts/DailyDialog/multi_gpu_train.sh
```

You can fine-tune PLATO w/o latent on different `${DATASET}`. We provide an example script on DailyDialog dataset.
```bash
sh scripts/DailyDialog/baseline_train.sh
```

#### Recommended settings

For the fine-tuning of our pre-trained model, it usually requires about 10 epochs to reach convergence with learning rate = 1e-5 and about 2-3 epochs to reach convergence with learning rate = 5e-5.

GPU Memory | batch size | max len
------|------|------
16G | 6 | 256
32G | 12 | 256

### Infer
Running inference on test dataset.
```bash
# DailyDialog / PersonaChat / DSTC7_AVSD
DATASET=DailyDialog
sh scripts/${DATASET}/infer.sh

# Running inference of PLATO w/o latent
sh scripts/DailyDialog/baseline_infer.sh
```
After inference, you can find the output foler `outputs/${DATASET}.infer` (by default). It contains `infer_0.result.json` (the inference result), `hparams.json` (hyper-parameters of inference scipt) and `trainer.log` (inference log).

If you want to use top-k sampling (beam search by default), you can follow the example script:
```bash
sh scripts/DailyDialog/topk_infer.sh
```

## Result

### DailyDialog
Model | BLEU-1/2 | Distinct-1/2 | Fluency | Coherence | Informativeness | Overall
------|------|------|------|------|------|-------
Seq2Seq | 0.336/0.268 | 0.030/0.128 | 1.85 | 0.37 | 0.44 | 0.33
iVAE_MI | 0.309/0.249 | 0.029/0.250 | 1.53 | 0.34 | 0.59 | 0.30
PLATO w/o Latent | 0.405/0.322 | 0.046/0.246 | 1.91 | **1.58** | 1.03 | 1.44
PLATO | 0.397/0.311 | **0.053/0.291** | **1.97** | 1.57 | **1.23** | **1.48**
ProphetNet | 0.444/0.392 | 0.039/0.211 |  |  |  | 
ProphetNet with twitter | 0.445/0.391 | 0.038/0.207 |  |  |  | 
ProphetNet reddit mask span | 0.461/0.402 | 0.038/0.208 |  |  |  | 
ProphetNet reddit mask last 20% turn | 0.460/0.401 | 0.038/0.207 |  |  |  | 
ProphetNet reddit mask random 20% turn | **0.461/0.403** | 0.039/0.227 |  |  |  | 
ProphetNet reddit mask random 20% turn (add turn level embedding) | **0.461/0.402** | 0.041/0.230 |  |  |  | 
seq2seq VAE (latent = 16, training...) | 0.424/0.346 | 0.046/0.283 |  |  |  | 
seq2seq VAE (latent = 32, training...) | 0.410/0.324 | 0.056/0.298  |  |  | 

### PersonaChat
Model | BLEU-1/2 | Distinct-1/2 | Knowledge R/P/F1 | Fluency | Coherence | Informativeness | Overall
------|------|------|------|------|------|-------|-------
Seq2Seq | 0.448/0.353 | 0.004/0.016 | 0.004/0.016/0.006 | 1.82 | 0.37 | 0.85 | 0.34
LIC | 0.405/0.320 | 0.019/0.113 | 0.042/0.154/0.064 | 1.95 | 1.34 | 1.09 | 1.29
PLATO w/o Latent | 0.458/0.357 | 0.012/0.064 | 0.085/0.263/0.125 | 1.98 | 1.36 | 1.04 | 1.30
PLATO | 0.406/0.315 | **0.021/0.121** | **0.142/0.461/0.211** | **1.99** | **1.51** | **1.70** | **1.50**
ProphetNet | 0.466/0.391 | 0.013/0.075 | 0.083/0.278/0.124 |  |  |  | 
ProphetNet with twitter | 0.468/0.392 | 0.013/0.075 | 0.086/0.281/0.129 |  |  |  | 
ProphetNet reddit mask span | 0.461/0.389 | 0.011/0.069 | 0.094/0.325/0.144 |  |  |  | 
ProphetNet reddit mask last 20% turn | **0.469/0.395** | 0.012/0.073 | 0.085/0.280/0.127 |  |  |  | 
ProphetNet reddit mask random 20% turn | 0.460/0.384 | 0.014/0.078 | 0.086/0.281/0.129 |  |  |  | 
ProphetNet reddit mask random 20% turn (add turn level embedding) | 0.461/0.385 | 0.012/0.070 | 0.101/0.345/0.151 |  |  |  | 
ProphetNet (drop knowledge) | 0.396/0.353 | 0.016/0.082 | 0.015/0.069/0.0244 |  |  |  | 
seq2seq VAE (latent = 16, training...) | **0.434/0.365** | 0.016/0.097 |  |  |  | 
seq2seq VAE (latent = 32, training...) | **0.419/0.331** | 0.020/0.109 |  |  |  | 

### DSTC7_AVSD
Model | BELU-1 | BELU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGH-L | CIDEr
------|------|------|------|------|------|-------|-------
Baseline | 0.629 | 0.485 | 0.383 | 0.309 | 0.215 | 0.487 | 0.746
CMU | 0.718 | 0.584 | 0.478 | 0.394 | 0.267 | 0.563 | 1.094
PLATO | 0.784 | 0.637 | 0.525 | 0.435 | 0.286 | 0.596 | 1.209
ProphetNet | 0.824 | 0.691 | 0.582 | 0.487 | 0.313 | 0.640 | 1.382
ProphetNet with twitter | 0.815 | 0.681 | 0.571 | 0.476 | 0.308 | 0.635 | 1.360
ProphetNet reddit mask span | 0.823 | 0.688 | 0.577 | 0.482 | 0.309 | 0.636 | 1.358
ProphetNet reddit mask last 20% turn | 0.829 | 0.702 | 0.593 | 0.498 | 0.311 | 0.641 | 1.382
ProphetNet reddit mask random 20% turn | 0.832 | 0.705 | 0.598 | 0.506 | 0.314 | 0.638 | 1.386
ProphetNet reddit mask random 20% turn (add turn level embedding) | **0.833** | **0.705** | **0.598** | **0.506** | **0.315** | **0.640** | **1.386**
encoder with ape and rpe (0.2 epoch) | **0.824** | **0.701** | **0.582** | **0.499** | **0.310** | **0.638** | **1.384**


Note: In the experiments on `DSTC7_AVSD`, the response selection of our method is strengthened with an extra ranking step, which ranks the candidates according to the automatic scores and selects the top one as the final answer.

## Citation
If you find PLATO useful in your work, please cite the following paper:
```
@inproceedings{bao2019plato,
    title={PLATO: Pre-trained Dialogue Generation Model with Discrete Latent Variable},
    author={Bao, Siqi and He, Huang and Wang, Fan and Wu, Hua and Wang, Haifeng},
    booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    pages={85--96},
    year={2020}
}
```

## Disclaimer
This project aims to facilitate further research progress in dialogue generation. Baidu is not responsible for the 3rd party's generation with the pre-trained system.

## Contact information
For help or issues using PLATO, please submit a GitHub issue.

For personal communication related to PLATO, please contact Siqi Bao (`baosiqi@baidu.com`), or Huang He (`hehuang@baidu.com`).
