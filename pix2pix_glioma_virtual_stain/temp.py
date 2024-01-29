import numpy as np
import os

import torch
import torchvision.transforms as transforms
import PIL.Image


# for root, dir, files in os.walk("/public/home/win0701/dataset/pair_cancer_npy_new/test_A"):
#     print(root)
#     print(dir)
#     print(files)
#     for file in files:
#         path = os.path.join(root, file)
#         print(path)
#         b = np.load(path).transpose((1, 2, 0))
#         np.save(path, b)



"""
    指定坐标位置裁剪图片

"""
import os
import cv2
import numpy as np

IMG_EXTENSIONS = [ "npy", "tif"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    name = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                name.append(fname)

    return images, name


def cut_cl(data, save_dir, fname, line=2, column=2, ):
    H, W, C = data.shape
    h_stride = H//line
    w_stride = W//column
    n = 0
    for i in range(line):
        for j in range(column):
            n += 1
            save_file = os.path.join(save_dir, fname + "_" + str(n) + ".tif")
            print("save file name:", save_file)
            # np.save(save_file, data[h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1), :])
            cv2.imwrite(save_file, data[h_stride*i:h_stride*(i+1), w_stride*j:w_stride*(j+1), :])

if __name__ == "__main__":
    np_path, np_name = make_dataset(r"/public/home/win0701/dataset/pair_cancer_npy_new/test_B")
    save_dir = r"/public/home/win0701/dataset/pair_cancer_npy_new/test_B"
    print("img_paths: ", np_path)
    for i in range(len(np_name)):
        # data = np.load(np_path[i])
        data = cv2.imread(np_path[i])
        cut_cl(data, save_dir, np_name[i][:-4])
        print("cutting: ", i)




#
# from collections import OrderedDict
# visuals = OrderedDict([('input_label', 1),
#                        ('synthesized_image', 2),
#                        ('ground_true_image', 3)])
#
# print(visuals.items())