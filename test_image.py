"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict

import data
from util import util
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from data.base_dataset import BaseDataset, get_params, get_transform

#Input-output
label_path = 'input/test_val_00000002.png'
generated_image_path = "results/generated.png"

#Loading the trained model
opt = TestOptions().parse()
model = Pix2PixModel(opt)
model.eval()

#Loading Semantic Label
label = Image.open(label_path)
params = get_params(opt, label.size)
transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
label_tensor = transform_label(label) * 255.0
label_tensor[label_tensor == 255] = opt.label_nc
print("-- Label tensor :", np.shape(label_tensor))

#Creating data_dictionay
data_i = {}
data_i['label'] = label_tensor.unsqueeze(0)
data_i['path'] = [None]
data_i['image'] = [None]
data_i['instance'] = [None]

#Inference code
generated = model(data_i, mode='inference')
for b in range(generated.shape[0]):
    generated_image = generated[b]
    generated_image = util.tensor2im(generated_image)
    generated_image_path_ = generated_image_path[:-4]+str(b)+".png"
    print('---- generated image ', generated_image_path_, np.shape(generated_image))
    cv2.imwrite(generated_image_path_, generated_image)