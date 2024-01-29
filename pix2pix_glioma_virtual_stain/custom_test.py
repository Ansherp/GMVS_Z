import os

import cv2
from PIL import Image
import util.util as util
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import torchvision as tv
import torch
from torch.autograd import Variable

opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

model = create_model(opt)
# net = torch.nn.DataParallel(model)
# net = net.to('gpu')

for j in range(100):
    opt.which_epoch = (j+1)*10
    model.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, "label2city", opt.which_epoch + "_net_G.pth")), False)

    for i, data in enumerate(dataset):
        out_img = model.inference(data['label'], data['inst'], data['image'])
        print(data["path"])
        print(out_img.shape)
        print(os.path.split(data["path"][0])[-1][:-4] + '.tif')
        out_img = util.tensor2im(out_img.data[0])
        cv2.imwrite(os.path.join(opt.results_dir, str(opt.which_epoch), os.path.split(data["path"][0])[-1][:-4] + '.png'), out_img)
