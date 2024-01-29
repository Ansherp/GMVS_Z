import os.path
import random

from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from PIL import ImageFile
import scipy.io as scio
import h5py
ImageFile.LOAD_TRUNCATED_IMAGES = True




class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        # opt.phase='test'      ############ 改这里
        ### input A (label maps)
        dir_A = 'A' if self.opt.label_nc == 0 else '_label'
        if opt.dataroot_custom:
            self.dir_A = opt.dataroot_A
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = 'B' if self.opt.label_nc == 0 else '_img'
            if opt.dataroot_custom:
                self.dir_B = opt.dataroot_B
            else:
                self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        B_path = []
        # print("pre  A_path", A_path)
        if self.opt.datatype_mat_n >= 0:
            try:
                A = h5py.File(A_path)
                A = A['AAA_unstained_image_all'][:]
                A = np.transpose(A, (2, 1, 0))
            except:
                A = scio.loadmat(A_path)
                A = A['AAA_unstained_image_all'][:]
            A = A[:,:, [self.opt.datatype_mat_n]]
            params = get_params(self.opt, A.shape)

        elif self.opt.npy_random:    # 加入随机抽取
            A = np.uint8(np.load(A_path))
            i_list = random.sample(range(A.shape[-1]), self.opt.input_nc)
            A = A[:, :, i_list]
            params = get_params(self.opt, A.shape)
        else:
            A = Image.open(A_path) if self.opt.input_nc <= 3 else np.uint8(np.load(A_path))
            params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB') if self.opt.input_nc == 3 else A)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path,  'path_B': B_path}

        # print(input_dict["path"], input_dict["path_B"])
        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'