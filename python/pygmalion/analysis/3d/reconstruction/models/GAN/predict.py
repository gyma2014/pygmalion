from copy import deepcopy
import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.backends.cudnn.benchmark = True

from skimage.transform import resize
gen = Generator(in_channels=3, features=64).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

models = r"C:\Users\gyma2\Documents\git\pygmalion\python\pygmalion\analysis\3d\reconstruction\models\GAN\recons_models"

import os
chk = os.path.join(models, r"110.gen.pth.tar")

load_checkpoint(chk, gen, opt_gen, config.LEARNING_RATE)

def tensor2np(tensor):
    grid = make_grid(tensor * 0.5 + 0.5)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr


def predict(image, out_file):

    image = image.to(config.DEVICE).unsqueeze(0)
    gen.eval()

    with torch.no_grad():
        Y = gen(image)

        save_image(Y*0.5 + 0.5, out_file)


in_file = r"C:\Users\gyma2\Documents\data\test_images\1233.jpg"

from PIL import Image
import numpy as np

im = np.array(Image.open(in_file))
# im = im[:, 0: 255, :]

augmentations = config.both_transform(image=im, image0=im)
input_image = augmentations["image"]

input_image = config.transform_only_input(image=input_image)["image"]

original = tensor2np(input_image)

predict(input_image, "./test.jpg")

import enum
from PIL import Image


p = "./test.jpg"

im = Image.open(p)

import numpy as np
arr = np.asarray(im)

depth = arr

import matplotlib.pyplot as plt
plt.imshow(depth)
plt.show()

# guassian smooth
import cv2

gray = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
_, gray = cv2.threshold(gray, 100, 255, cv2.CV_8UC1)

# find contours
kernel = np.ones((5, 5), 'uint8')
gray = cv2.erode(gray, kernel, iterations=1)
contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


largebox = None

if len(contours) > 0:
    largebox = cv2.boundingRect(contours[0])

contours = [c for c in contours if 3 * 3 < cv2.contourArea(c) <= 50*50]

depth_copy = deepcopy(depth)
#depth = cv2.GaussianBlur(depth, (5, 5), cv2.BORDER_DEFAULT)

for c in contours:
    box = cv2.boundingRect(c)
    depth[box[1]:box[1]+box[3], box[0]: box[0]+box[2]] = depth_copy[box[1]:box[1]+box[3], box[0]: box[0]+box[2]]



def dumpOff(arr, delta_x, delta_y, height, image, file):

    points = []

    map = {}

    def is_valid(x, y, arr):
      width = len(arr[0])
      height = len(arr)

      return x > 2 and x < width - 2 and y > 2 and y < height - 2 and arr[y][x][0] > 20

    def neighbors(y, x, arr):
        eight_points = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        ns = []

        for dy, dx in eight_points:
            if is_valid(x+dx, y+dy, arr):
                ns.append(arr[y+dy][x+dx][0])

        return ns

    for y, row in enumerate(arr):
        for x, z in enumerate(row):
            ns = neighbors(y, x, arr)
            # we want to:
            # 0. has valid neighbors
            if not is_valid(x, y, arr):
                continue

            if len(ns) == 0:
                z = 0
            else:
                mmax = max(ns)
                mmin = min(ns)
                import statistics
                mmean = int(statistics.mean(ns))

                if z[0] > mmax + 2 or z[0] < mmin - 2:
                    z[0] = mmean # desnoise
                else:
                    #z[0] = (mmean * len(ns) + z[0])/(len(ns)+1) # smooth
                    z[0] = mmean/3 + mmean/3 + z[0]/3

    for y, row in enumerate(arr):
        for x, z in enumerate(row):
          if is_valid(x, y, arr):
            points.append(
                f'{x-delta_x} {height-y+delta_y} {arr[y][x][0]}')
            map[(x, y)] = len(points)-1

    edges = []

    for y in range(len(arr)):
        for x in range(len(arr[y])):
            if (x, y) in map and (x+1, y) in map and (x, y+1) in map and (x+1, y+1) in map:
                tl = map[(x, y)]
                tr = map[(x+1, y)]
                bl = map[(x, y+1)]
                br = map[(x+1, y+1)]

                # skip trianglation if max-min is larger than 10
                minz = min([arr[y][x][0], arr[y+1][x][0], arr[y+1][x+1][0], arr[y][x+1][0]])
                maxz = max([arr[y][x][0], arr[y+1][x][0], arr[y+1][x+1][0], arr[y][x+1][0]])

                if maxz-minz >= 15:
                    continue

                edges.append(f'3 {tr} {tl} {bl} {image[y][x][0]} {image[y][x][1]} {image[y][x][2]} 1')
                edges.append(f'3 {bl} {br} {tr} {image[y][x][0]} {image[y][x][1]} {image[y][x][2]} 1')


    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(f"{len(points)} {len(edges)} {0}\n")
        fp.write('\n'.join(points))
        fp.write('\n')
        fp.write('\n'.join(edges))

dumpOff(depth, 0, 0, 256, original, "./test.off")
