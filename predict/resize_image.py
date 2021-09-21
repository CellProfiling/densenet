import os
import cv2
import numpy as np
from PIL import Image
import mlcrate as mlc
import argparse

opj = os.path.join
ope = os.path.exists

def do_resize(param):
    src, fname, dst, size = param
    # eg. 44741_1177_B2_3_red.jpg -> red
    color = fname[fname.rfind('_') + 1:fname.rfind('.')]
    try:
        image = np.array(Image.open(opj(src, fname)))[:, :, COLOR_INDEXS.get(color)]
    except:
        print('bad image : %s' % fname)
        image = cv2.imread(opj(src, fname))[:, :, -1::-1][:, :, COLOR_INDEXS.get(color)]
    h, w = image.shape[:2]
    if h != size or w != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(opj(dst, fname), image,  [int(cv2.IMWRITE_JPEG_QUALITY), 85])

COLOR_INDEXS = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'yellow': 0,
}

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--src_dir', type=str, default=None, help='source directory')
parser.add_argument('--dst_dir', type=str, default=None, help='destination directory')
parser.add_argument('--size', type=int, default=1536, help='size')
args = parser.parse_args()

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    size = args.size
    src = args.src_dir
    dst = args.dst_dir
    dst = opj(dst, 'images_%d' % size)

    fnames = np.sort(os.listdir(src))
    os.makedirs(dst, exist_ok=True)

    start_num = max(0, len(os.listdir(dst)))
    fnames = fnames[start_num:]
    params = [(src, fname, dst, size) for fname in fnames]

    pool = mlc.SuperPool(10)
    pool.map(do_resize, params, description='resize image')

    print('success.')
