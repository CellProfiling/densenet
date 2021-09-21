"""A module designed to run the HPA Kaggle challenge winning model."""
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
import argparse

from dataset import *
from models import *

warnings.filterwarnings("ignore")


opj = os.path.join
ope = os.path.exists


class Config(object):
    """This class acts as the configuration for all model functions.

    This class essentially acts as a python dict with information for the
    running of model predictions.

    The following fields are in the class, and should be changed according
    to your system and preferences:

    gpus: A string of IDs to available Nvidia GPUs, separated by commas.
    num_workers: The number of threads that runs prediction, as an integer.
                 Note that the number of available GPUs may impact how
                 many threads you can safely run.
    batch_size: The size of each prediction batch run by the threads, as an integer.
                This should not be too large for your GPU to fit the images
                into memory.
    num_classes: The number of classes that the model predicts, as an integer.
                 The HPA Kaggle challenge winning model specifically requires 28 classes.
    image_size: The image size as an integer, in pixels, that will be used as input for the network.
                Each image should be <image_size>x<image_size> large.
                The HPA Kaggle challenge winning model should specifically use 1536 pixels here.
    crop_size: The image size as an integer, in pixels, that will be cropped from the input images.
                The HPA Kaggle challenge winning model should specifically use 1024 pixels here.
    in_channels: The number of input channels for the network, as an integer, usually 3 or 4.
                 When working with HPA data, use 3 for RBG and 4 for RGBY.
                 The HPA Kaggle challenge winning model should use 4 here.
    seeds: The random seeds to be used for random crops, as a list of integers.
           A value of [0] means no crop.
           To recreate the results from HPA papers using the HPA Kaggle challenge winning model,
           seeds should be equal to [0] (no crop).
    augments: A list of strings describing with test time augmentations to use.
              To recreate the results from HPA papers using the HPA Kaggle challenge winning model,
              seeds should be equal to ["default"].
    model_name: The name of the model to be used for predictions, as a string.
    model_path: A file path, as a string, to the pytorch model file.
    suffix: A string with the file ending of all input images, for example "jpg".
    image_dir: The input image directory, as a string, from which images will be taken to feed the model.
               Note that if you have used the `resize_image.py` script, this would be the folder
               with the suffix `_1536` in the `out_dir`.
    result_name: A string containing a name to be used as part of the output file names.
    out_dir: A file path, as a string, to a folder in which prediction results will be stored.
    features_dir: A file path, as a string, to a folder in which feature outputs will be stored.
    """

    gpus = "0,1"  # Follow the same convention as CUDA_VISIBLE_DEVICES
    num_workers = 2
    batch_size = 32

    num_classes = 28
    image_size = 1536
    crop_size = 1024
    in_channels = 4

    seeds = [0]
    augments = ["default"]
    model_name = "class_densenet121_large_dropout"

    model_path = "../models/model.pth"

    suffix = "jpg"

    image_dir = "../resized_images/images_1536"
    result_name = "example"
    out_dir = "../results/"
    features_dir = out_dir


def prob_to_result(probs, image_ids, th=0.5):
    probs = probs.copy()
    probs[range(len(probs)), np.argmax(probs, axis=1)] = 1

    pred_list = []
    for line in probs:
        s = " ".join(list([str(i) for i in np.nonzero(line > th)[0]]))
        pred_list.append(s)
    result_df = pd.DataFrame({ID: image_ids, PREDICTED: pred_list})
    return result_df


def predict_augment(model, dataloader, augment, seed, opt):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # eg. augment_default
    transform = eval("augment_%s" % augment)
    dataloader.dataset.set_transform(transform=transform)
    random_crop = (opt.crop_size > 0) and (seed != 0)
    dataloader.dataset.set_random_crop(random_crop=random_crop)

    out_dir = opt.out_dir
    features_dir = opt.features_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    print("out dir: %s **%s**" % (out_dir, opt.result_name))

    result_csv_fname = opj(out_dir, "results_%s.csv.gz" % opt.result_name)
    result_prob_fname = opj(out_dir, "probs_%s.npy" % opt.result_name)
    result_feature_fname = opj(features_dir, "features_%s.npz" % opt.result_name)

    prob_list = []
    feature_list = []
    image_ids = np.array(dataloader.dataset.image_ids)
    for iter, (images, indices) in tqdm(
        enumerate(dataloader, 0), total=len(dataloader)
    ):
        with torch.no_grad():
            images = Variable(images.cuda())
            logits, features = model(images)

            # probabilities
            probs = F.sigmoid(logits)
            prob_list += probs.cpu().data.numpy().tolist()

            # features
            features = features.cpu().data.numpy().tolist()
            feature_list.extend(features)

    # save result
    probs = np.array(prob_list)
    print("result prob shape: %s" % str(probs.shape))
    np.save(result_prob_fname, probs)

    features = np.array(feature_list, dtype="float32")
    print("result feature shape: %s" % str(features.shape))
    np.savez_compressed(result_feature_fname, feats=features)

    result_df = prob_to_result(probs, image_ids)
    print("result csv shape: %s" % str(result_df.shape))
    print(result_df.head())
    result_df.to_csv(result_csv_fname, index=False, compression="gzip")
    return probs, features


def do_predict(model, dataloader, opt):
    model = model.eval()
    results = []
    for seed in opt.seeds:
        for augment in opt.augments:
            ret = predict_augment(model, dataloader, augment, seed, opt)
            results.append(ret)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict densenet 121')
    parser.add_argument('--gpus', type=str, default='', help='GPU device numbers, e.g. 0,1')
    parser.add_argument('--image_dir', type=str, default='../resized_images/images_1536', help='input image folder')
    parser.add_argument('--out_dir', type=str, default='../results/', help='output image folder')

    args = parser.parse_args()
    print("%s: calling main function ... " % os.path.basename(__file__))

    ID = "Id"
    PREDICTED = "Predicted"

    opt = Config()
    opt.image_dir = args.image_dir
    opt.gpus = args.gpus

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    print("use gpus: %s" % opt.gpus)

    model = eval(opt.model_name)(
        num_classes=opt.num_classes,
        in_channels=opt.in_channels,
        pretrained=opt.model_path,
    )
    model = DataParallel(model)
    model = model.cuda()
    print("model name: %s" % opt.model_name)

    dataset = ProteinDataset(
        opt.image_dir,
        image_size=opt.image_size,
        crop_size=opt.crop_size,
        in_channels=opt.in_channels,
        suffix=opt.suffix,
    )
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=opt.batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    print("len(dataset): %d" % len(dataset))

    do_predict(model, dataloader, opt)

    print("\nsucess!")
