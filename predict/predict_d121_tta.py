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

    gpus = "0,1,2,3"
    num_workers = 4
    batch_size = 32

    num_classes = 28
    image_size = 1536
    crop_size = 1024
    in_channels = 4

    seeds = [0]
    augments = [
        "default",
        "flipud",
        "fliplr",
        "transpose",
        "flipud_lr",
        "flipud_transpose",
        "fliplr_transpose",
        "flipud_lr_transpose",
    ]  # test time augmentation
    model_name = "class_densenet121_large_dropout"

    model_path = "../models/model.pth"

    suffix = "jpg"

    image_dir = "../images/images_1536"
    result_name = "resultname"
    out_dir = "../outputdirectory"
    features_dir = "../features/"


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

    prob_list = []
    image_ids = np.array(dataloader.dataset.image_ids)
    for iter, (images, indices) in tqdm(
        enumerate(dataloader, 0), total=len(dataloader)
    ):
        with torch.no_grad():
            images = Variable(images.cuda())
            logits, _ = model(images)

            # probabilities
            probs = F.sigmoid(logits)
            prob_list += probs.cpu().data.numpy().tolist()

    probs = np.array(prob_list)
    return image_ids, probs


def do_predict(model, dataloader, opt):
    model = model.eval()

    out_dir = opj(opt.out_dir, "maximum")
    os.makedirs(out_dir, exist_ok=True)
    print("out dir: %s **%s**" % (out_dir, opt.result_name))

    result_csv_fname = opj(out_dir, "results_%s.csv.gz" % opt.result_name)
    result_prob_fname = opj(out_dir, "probs_%s.npy" % opt.result_name)

    result_image_ids = None
    result_probs = None
    for seed in opt.seeds:
        for augment in opt.augments:
            image_ids, probs = predict_augment(model, dataloader, augment, seed, opt)

            if result_image_ids is None:
                result_image_ids = image_ids
                result_probs = probs
            else:
                assert np.array_equal(result_image_ids, image_ids)
                result_probs = np.dstack((result_probs, probs)).max(axis=-1)

    # save result
    print("result prob shape: %s" % str(result_probs.shape))
    np.save(result_prob_fname, result_probs)

    result_df = prob_to_result(result_probs, result_image_ids)
    print("result csv shape: %s" % str(result_df.shape))
    print(result_df.head())
    result_df.to_csv(result_csv_fname, index=False, compression="gzip")


if __name__ == "__main__":
    print("%s: calling main function ... " % os.path.basename(__file__))

    ID = "Id"
    PREDICTED = "Predicted"

    opt = Config()

    # gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = True
    print("use gpus: %s" % opt.gpus)

    # model
    model = eval(opt.model_name)(
        num_classes=opt.num_classes,
        in_channels=opt.in_channels,
        pretrained=opt.model_path,
    )
    model = DataParallel(model)
    model = model.cuda()
    print("model name: %s" % opt.model_name)

    # dataset
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
