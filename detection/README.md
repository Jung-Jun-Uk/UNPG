## K-FACE data prepration

K-FACE download link: [AI-hub](https://aihub.or.kr/).

Detail configuration about K-FACE is provided in the paper below.

[K-FACE: A Large-Scale KIST Face Database in Consideration with
Unconstrained Environments](https://arxiv.org/abs/2103.02211)

K-FACE sample images

![title](../image/kface_sample.png)

Structure of the K-FACE database

![title](../image/structure_of_kface.png)

### Detection & Alignment on K-FACE

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
cd kface_detection
python align_kfaces.py --ori_data_path '/data/ssd/jju/KFACE/High' --detected_data_path '/data/ssd/jju/KFACE/kface_retina_align_112x112
```
We referred to [https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface).

### Training and test datasets on K-FACE 
We already create the T1-T4 YAML file and Q1-Q4 .txt file. See the [KFACE]() forder
![title](../image/training_and_test_datasets_on_kface.PNG)