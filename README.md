## UNPG
Unified Negative Pair Generation toward
Well-discriminative Feature Space for Face
Recognition, arXiv, 2022

![](_images/geo_intro.png)
<p align="center">
<img src="_images/multi_vs_uni.png"  width="300" height="300"/>
</p>

## Data prepration

### [MS1MV2](https://arxiv.org/abs/1801.07698)
MS1M-ArcFace (85K ids/5.8M images) [download link](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

```bash
#Preprocess 'train.rec' and 'train.idx' to 'jpg'

# example
cd detection

python rec2image.py --include '{path}/face_emore' --output '{path}/MS1MV2'
```

### [K-FACE](https://arxiv.org/abs/2103.02211)
K-FACE [download link](https://github.com/Jung-Jun-Uk/mixface)

```bash
"""
    ###################################################################

    K-Face : Korean Facial Image AI Dataset
    url    : http://www.aihub.or.kr/aidata/73

    Directory structure : High-ID-Accessories-Lux-Emotion
    ID example          : '19062421' ... '19101513' len 400
    Accessories example : 'S001', 'S002' .. 'S006'  len 6
    Lux example         : 'L1', 'L2' .. 'L30'       len 30
    Emotion example     : 'E01', 'E02', 'E03'       len 3
    
    ###################################################################
"""

# example
cd detection

python align_kfaces.py --ori_data_path '/data/FACE/KFACE/High' --detected_data_path 'kface_retina_align_112x112'
```

### IJBB & IJBC
[download link](https://github.com/IrvingMeng/MagFace)

Please apply for permissions from [NIST](https://www.nist.gov/programs-projects/face-challenges) before your usage.

## Evaluation

### Pretrained Model

|Method|IJB-B(1e-4)|IJB-C(1e-4)|MegaFace (Rank-1 acc.)|LFW|Path|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Circle-loss*|-|93.95|98.50|99.73|-|
|ArcFace*|94.20|95.60|98.35|99.82|-|
|MagFace*|94.51|95.97|-|**99.83**|-|
|CosFace|94.39|96.35|99.08|**99.83**|-|
|CosFace+UNPG|94.99|**96.39**|**99.27**|99.81|[link](https://koreatechackr-my.sharepoint.com/:u:/g/personal/rnans33_koreatech_ac_kr/EfQrB42yUHlIpy_G-tg7UH4BedVBFywGVRivRTwzkzyeRQ?e=xJ6T48)|
|ArcFace|94.89|96.25|98.56|**99.83**|-|
|ArcFace+UNPG|95.04|96.33|98.82|**99.83**|[link](https://koreatechackr-my.sharepoint.com/:u:/g/personal/rnans33_koreatech_ac_kr/EZlqt0175BVFmG0VvsnhNc8Bym9e18BHt0mrsDXAuk9eMw?e=h75aWI)|
|MagFace|93.99|95.54|98.51|99.81|-|
|MagFace+UNPG|**95.21**|96.38|98.03|99.81|[link](https://koreatechackr-my.sharepoint.com/:u:/g/personal/rnans33_koreatech_ac_kr/EYPx3wZNc3xMkULR7RpIgK0BK0UY_iHs6oZnkg49Xm21sw)|


```bash
cd recognition

# example
python evaluation.py --weights 'face.r100.cos.unpg.wisk1.5.pt' --data 'ijbc.yaml' 
# --data (e.g., 'ijbb.yaml', 'bins.yaml', 'kface.yaml')
```

## Training
#### Example script (FACE)
```bash
cd recognition

# example 
python train.py --model 'iresnet-34' --head 'arcface' --unpg_wisk 1.0 --data 'data/face.yaml' --hyp 'data/hyp.yaml' --name 'example' --device 0,1
```

#### Example script (KFACE)
```bash
cd recognition

# example 
python train.py --model 'iresnet-34' --head 'arcface' --unpg_wisk 1.0 --data 'data/kface.yaml' --hyp 'data/hyp.yaml' --name 'example' --device 0,1
```