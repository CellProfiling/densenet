0.Environment and file descriptions
0.1 The basic Runtime Environment is python3.6, pytorch0.4.1, you can refer to requriements.txt to set up your environment.

1. Prepare data
1.1 Please config your local directory in class Config (including predict.py and predict_metric_learning.py).

2. Preprocess data
2.1 Resize v18 external image
process :
    cd CODE_DIR/
    python resize_image.py --src_dir ${src_dir} --dst_dir --${dst_dir} --size ${size}
input :
    ${src_dir}/
output :
    ${dst_dir}/images_${size}

3. Predict
3.1 Predict densenet121 fold0 seed0, image size:1536x1536, crop size:1024x1024, augment:default
seed0 means don't cropping the image
process :
    cd CODE_DIR/
    python predict_d121.py
input :
    image_dir/
    model_path
output :
    out_dir/default_seed0/results_test.csv.gz
    out_dir/default_seed0/probs_test.npy
    out_dir/default_seed0/features_test.npz

3.2 Predict densenet121 fold0 tta, image size:1536x1536, crop size:1024x1024
process :
    cd CODE_DIR/
    python predict_d121_tta.py
input :
    image_dir/
    model_path
output :
    out_dir/maximum/results_test.csv.gz
    out_dir/maximum/probs_test.npy

3.3 Predict resnet50 arcface model, image size:768x768, augment:default
process :
    cd CODE_DIR/
    python predict_r50_metric_learning.py
input :
    image_dir/
    model_path
output :
    out_dir/default/results_test.csv.gz
    out_dir/default/features_test.npz
