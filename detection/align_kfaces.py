import os
import cv2 as cv
import numpy as np

import argparse
from alignment import warp_and_crop_face, get_reference_facial_points
from retinaface.detector import RetinafaceDetector


"""
    ###################################################################

    K-Face : Korean Facial Image AI Training Dataset
    url    : http://www.aihub.or.kr/aidata/73

    Directory structure : High-ID-Accessories-Lux-Emotion
    ID example          : '19062421' ... '19101513' len 400
    Accessories example : 'S001', 'S002' .. 'S006'  len 6
    Lux example         : 'L1', 'L2' .. 'L30'       len 30
    Emotion example     : 'E01', 'E02', 'E03'       len 3
    S001 - L1, every emotion folder contaions a information txt file
    (ex. bbox, facial landmark) 
    
    ###################################################################
"""


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def create_aligned_kface_dataset(ori_data_path, 
                                 copy_data_path,
                                 detector,
                                 output_img_size=(112,112)):

    accessories = ['S' + str(a).zfill(3) for a in range(1, 6 + 1)]
    luces = ['L' + str(l) for l in range(1, 30 + 1)]
    expressions = ['E' + str(e).zfill(2) for e in range(1,3 + 1)]
    poses = ['C' + str(p) for p in range(1, 20 + 1)]

    print('Aceessories : ',accessories)
    print('Lux         : ',luces)
    print('Expression  : ',expressions)
    print('Pose        : ',poses)

    identity_lst = os.listdir(ori_data_path)
    for i, idx in enumerate(identity_lst):
        print(i+1, "preprocessing..")
        for p in poses:
            for a in accessories:
                for l in luces:
                    for e in expressions:
                        ori_image_path = os.path.join(ori_data_path, idx, a, l, e, p) + '.jpg'
                        copy_dir = os.path.join(copy_data_path, idx, a, l, e)
                        mkdir_if_missing(copy_dir)
                        
                        copy_image_path = os.path.join(copy_dir, p) + '.jpg'
                        raw = cv.imread(ori_image_path)
                        if a == 'S001' and l == 'L1' and e == 'E01':
                            _, facial5points = detector.detect_faces(raw)
                            facial5points = np.reshape(facial5points[0], (2, 5))
                        
                        default_square = True
                        inner_padding_factor = 0.25
                        outer_padding = (0, 0)

                        # get the reference 5 landmarks position in the crop settings
                        reference_5pts = get_reference_facial_points(
                            output_img_size, inner_padding_factor, outer_padding, default_square)

                        # dst_img = warp_and_crop_face(raw, facial5points, reference_5pts, crop_size)
                        dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=output_img_size)
                        cv.imwrite(copy_image_path, dst_img)
        

def parser():   
    parser = argparse.ArgumentParser(description='KFACE detection and alignment')
    parser.add_argument('--ori_data_path', type=str, default='/data/data_server/jju/datasets/FACE/kface-retinaface-112x112', help='raw KFACE path')
    parser.add_argument('--detected_data_path', type=str, default='kface-retinaface-test', help='output path')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    opt = parser()    
    detector = RetinafaceDetector()        
    create_aligned_kface_dataset(ori_data_path=opt.ori_data_path, 
                                 copy_data_path=opt.detected_data_path,
                                 detector=detector,
                                 output_img_size=(112,112))