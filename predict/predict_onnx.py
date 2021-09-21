"""
This file runs the exported onnx model for prediction

It will automatically download the model and images.
One difference compared to the original pytorch model is that it produce class probabilities instead of logits.

Author: Wei OUYANG
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
import urllib.request

COLORS =  ["red", "green", "blue", "yellow"]
LABELS = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',
14:  'Microtubules',
15:  'Microtubule ends',
16:  'Cytokinetic bridge',
17:  'Mitotic spindle',
18:  'Microtubule organizing center',
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',
22:  'Cell junctions',
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',
27:  'Rods & rings' }

def read_rgby(work_dir, image_id, crop_size=1024, suffix='jpg'):
    image = [
        cv2.imread(
            os.path.join(work_dir, "%s_%s.%s" % (image_id, color, suffix)),
            cv2.IMREAD_GRAYSCALE,
        )
        for color in COLORS
    ]
    image = np.stack(image, axis=-1)
    h, w = image.shape[:2]
    if crop_size != h or crop_size != w:
        image = cv2.resize(
            image,
            (crop_size, crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
    return image

def fetch_image(work_dir, img_id):
    v18_url = 'http://v18.proteinatlas.org/images/'
    img_id_list = img_id.split('_')
    for color in COLORS:
        img_url = v18_url + img_id_list[0] + '/' + '_'.join(img_id_list[1:]) + '_' + color + '.jpg'
        img_name = img_id + '_' + color + '.jpg'
        fpath = os.path.join(work_dir, img_name)
        if not os.path.exists(fpath):
            urllib.request.urlretrieve(img_url, fpath)

def predict_hpa_images(
    image_ids,
    work_dir = './data',
    threshold = 0.5,
    model_path = None
    ):
    os.makedirs(work_dir, exist_ok=True)
    model_path = model_path or os.path.join(work_dir, 'densenet_model_probability.onnx')
    if not os.path.exists(model_path):
        urllib.request.urlretrieve('https://github.com/CellProfiling/densenet/releases/download/v0.1.0/external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds_fold0_final_probability.onnx', model_path)

    ort_session = ort.InferenceSession(model_path)
    for image_id in image_ids:
        fetch_image(work_dir, image_id)
        input_image = read_rgby(work_dir, image_id).astype('float32')
        input_image = input_image.transpose(2, 0, 1)
        input_image = input_image.reshape(1, 4, input_image.shape[1], input_image.shape[2])
        classes, features  = ort_session.run(None, {'image': input_image / 255.0})
        pred = [(LABELS[i], prob) for i, prob in enumerate(classes[0].tolist()) if prob>threshold]
        print('Image:', image_id, 'Prediction:', pred, 'Features:', features)

if __name__ == '__main__':
    image_ids = ['115_672_E2_1']
    predict_hpa_images(image_ids)

