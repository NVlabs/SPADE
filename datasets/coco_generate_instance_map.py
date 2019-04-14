"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import argparse
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from skimage.draw import polygon

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_file', type=str, default="./annotations/instances_train2017.json",
                    help="Path to the annocation file. It can be downloaded at http://images.cocodataset.org/annotations/annotations_trainval2017.zip. Should be either instances_train2017.json or instances_val2017.json")
parser.add_argument('--input_label_dir', type=str, default="./train_label/",
                    help="Path to the directory containing label maps. It can be downloaded at http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip")
parser.add_argument('--output_instance_dir', type=str, default="./train_inst/",
                    help="Path to the output directory of instance maps")

opt = parser.parse_args()

print("annotation file at {}".format(opt.annotation_file))
print("input label maps at {}".format(opt.input_label_dir))
print("output dir at {}".format(opt.output_instance_dir))

# initialize COCO api for instance annotations
coco = COCO(opt.annotation_file)


# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
imgIds = coco.getImgIds(catIds=coco.getCatIds(cats))
for ix, id in enumerate(imgIds):
    if ix % 50 == 0:
        print("{} / {}".format(ix, len(imgIds)))
    img_dict = coco.loadImgs(id)[0]
    filename = img_dict["file_name"].replace("jpg", "png")
    label_name = os.path.join(opt.input_label_dir, filename)
    inst_name = os.path.join(opt.output_instance_dir, filename)
    img = io.imread(label_name, as_grey=True)

    annIds = coco.getAnnIds(imgIds=id, catIds=[], iscrowd=None)
    anns = coco.loadAnns(annIds)
    count = 0
    for ann in anns:
        if type(ann["segmentation"]) == list:
            if "segmentation" in ann:
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    rr, cc = polygon(poly[:, 1] - 1, poly[:, 0] - 1)
                    img[rr, cc] = count
                count += 1

    io.imsave(inst_name, img)
