'''
class_stats for both conditioned on include and exclude.
'''

from copy import deepcopy
import os
import random

import MinkowskiEngine as ME
import numpy as np
import torch
import yaml
import pickle

from numba import jit
from tqdm import tqdm
from utils.voxelizer import Voxelizer
from utils import unkn_labels as unk_labels

import multiprocessing as mp
import pdb
# from utils.dataset import SemanticKITTIRestrictedDisjointDataset

class SemanticKITTIRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            inverse_map,
            t,
        )
    
    def get_raw_data(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            # if self.label_mapping_function is not None:
            #     mapped_labels = self.label_mapping_function(labels)
            # else:
            #     mapped_labels = np.copy(labels)
        
        return (
            coordinates,
            features,
            # mapped_labels,
            selected_idx,
            labels,
            t,
        )


class SemanticPOSSRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/semposs_dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        learning_map = self.config["learning_map"]
        self.learning_map_function = np.vectorize(lambda x: learning_map[x])

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))
        # features = np.expand_dims(np.divide(remissions, 255), 1)
        # if self.augment:
        #     random_noise = (np.random.rand(features.shape[0], 1) - 0.5) * 0.1
        #     features += random_noise
        #     features = np.clip(features, 0, 1)

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            labels = self.learning_map_function(labels)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            inverse_map,
            t,
        )
    
    def get_raw_data(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))
        # features = np.expand_dims(np.divide(remissions, 255), 1)
        # if self.augment:
        #     random_noise = (np.random.rand(features.shape[0], 1) - 0.5) * 0.1
        #     features += random_noise
        #     features = np.clip(features, 0, 1)

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            labels = self.learning_map_function(labels)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            # if self.label_mapping_function is not None:
            #     mapped_labels = self.label_mapping_function(labels)
            # else:
            #     mapped_labels = np.copy(labels)

        # discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
        #     coordinates=coordinates,
        #     return_index=True,
        #     return_inverse=True,
        #     quantization_size=self.voxel_size,
        # )

        # unique_feats = features[unique_map]
        # unique_labels = labels[unique_map]
        # unique_mapped_labels = mapped_labels[unique_map]
        # selected_idx = selected_idx[unique_map]

        # return (
        #     discrete_coords,
        #     unique_feats,
        #     unique_labels,
        #     selected_idx,
        #     unique_mapped_labels,
        #     inverse_map,
        #     t,
        # )
        return (
            coordinates,
            features,
            selected_idx,
            labels,
            t,
        )


class nuScenesRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
        return_ref=False,
        nusc=None,
    ):
        self.return_ref = return_ref
        
        with open(config_file, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.data_path = nuscenesyaml['dataset_path']
        
        if split == 'train':
            self.imageset = os.path.join(self.data_path, 'origin_nuscenes_infos_train.pkl')
        elif split == 'valid':
            self.imageset = os.path.join(self.data_path, 'origin_nuscenes_infos_val.pkl')
        elif split == 'test':
            self.imageset = os.path.join(self.data_path, 'origin_nuscenes_infos_test.pkl')
        
        with open(self.imageset, 'rb') as f:
            data = pickle.load(f)

        # if split == 'train' or split == 'valid':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True)
        # elif split == 'test':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-test', dataroot=self.data_path, verbose=True)
        self.nusc_infos = data['infos']
        self.nusc = nusc
        
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )
            
        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        if self.imageset.find('test') != -1:
            raise NotImplementedError
            points_label = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
        else:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

        coordinates = points[:, :3]
        remissions = points[:, 3] # 0~255
        
        features = np.ones((coordinates.shape[0], 1))
        
        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            downsampling = min(self.downsampling, coordinates.shape[0])
            selected_idx = np.random.choice(
                coordinates.shape[0], downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])
        
        labels = points_label.squeeze(1)
        
        if self.imageset.find('test') == -1:
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
        
        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )
        
        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        
        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            inverse_map,
            index,
        )
    
    def get_raw_data(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        if self.imageset.find('test') != -1:
            raise NotImplementedError
            points_label = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
        else:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

        coordinates = points[:, :3]
        remissions = points[:, 3] # 0~255
        
        features = np.ones((coordinates.shape[0], 1))
        
        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            downsampling = min(self.downsampling, coordinates.shape[0])
            selected_idx = np.random.choice(
                coordinates.shape[0], downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])
        
        labels = points_label.squeeze(1)
        
        if self.imageset.find('test') == -1:
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
        
        # discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
        #     coordinates=coordinates,
        #     return_index=True,
        #     return_inverse=True,
        #     quantization_size=self.voxel_size,
        # )
        
        # unique_feats = features[unique_map]
        # unique_labels = labels[unique_map]
        # unique_mapped_labels = mapped_labels[unique_map]
        # selected_idx = selected_idx[unique_map]
        
        return (
            coordinates,
            features,
            selected_idx,
            labels,
            index,
        )


@jit('b1[:](b1[:],u1[:])', nopython=True, cache=True, parallel=False)
def nb_function(bool_mat, labels):
    unq_labels = np.unique(labels)
    bool_mat[unq_labels] = True
    return bool_mat
        
if __name__ == "__main__":
    test = False
    # dataset = 'nusc'
    dataset = 'kitti'
    # data_split = 'train'
    data_split = 'valid'
    assert dataset in ['poss', 'kitti', 'kitti-fine', 'kitti-finer', 'nusc']
    assert data_split in ['train', 'valid']
    if test:
        dataset_config_path = 'config/semkitti_dataset.yaml'
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        voxel_size = 0.05
        downsampling = 60000
        
        unknown_labels = []
        number_of_unk=len(unknown_labels)
        
        label_mapping, label_mapping_inv, unknown_label = unk_labels.label_mapping(
            unknown_labels, dataset_config['learning_map_inv'].keys())
        
        data_idx_file_path = 'selected_idx_2_3_4_5_6.npy'
        dataset = SemanticKITTIRestrictedDisjointDataset(
                config_file=dataset_config_path,
                data_idx_file=data_idx_file_path,
                split="train",
                voxel_size=voxel_size,
                downsampling=downsampling,
                augment=False,
                label_mapping=label_mapping,
            )
        
        classes = data_idx_file_path.split('selected_idx_')[1].split('.npy')[0].split('_')
        classes = [int(cls) for cls in classes]
            
        for i in tqdm(range(len(dataset))):
            coordinates, features, mapped_labels, selected_idx, labels, inverse_map, t = dataset.__getitem__(i)

            unq_labels = np.unique(labels)
            assert not np.isin(unq_labels, classes).any(), "failed test!!! there exists selected labels."

        print('success test!!!')
        pdb.set_trace()
        # np.save('class_stats_test.npy', bool_matrix)
    
    flag = False
    if data_split == 'train':    
        if (os.path.isfile('class_stats.npy') and dataset=='kitti') or (os.path.isfile('class_stats_poss.npy') and dataset=='poss') \
            or (os.path.isfile('class_stats_fine.npy') and dataset=='kitti-fine') or (os.path.isfile('class_stats_finer.npy') and dataset=='kitti-finer') \
            or (os.path.isfile('class_stats_nusc.npy') and dataset=='nusc'):
            if dataset == 'kitti':
                bool_matrix = np.load('class_stats.npy')
            elif dataset == 'kitti-fine':
                bool_matrix = np.load('class_stats_fine.npy')
            elif dataset == 'kitti-finer':
                bool_matrix = np.load('class_stats_finer.npy')
            elif dataset == 'poss':
                bool_matrix = np.load('class_stats_poss.npy')
            elif dataset == 'nusc':
                bool_matrix = np.load('class_stats_nusc.npy')
            flag = True
    elif data_split == 'valid':
        if (os.path.isfile('class_stats_val.npy') and dataset=='kitti') or (os.path.isfile('class_stats_poss_val.npy') and dataset=='poss') \
            or (os.path.isfile('class_stats_fine_val.npy') and dataset=='kitti-fine') or (os.path.isfile('class_stats_finer_val.npy') and dataset=='kitti-finer') \
            or (os.path.isfile('class_stats_nusc_val.npy') and dataset=='nusc'):
            if dataset == 'kitti':
                bool_matrix = np.load('class_stats_val.npy')
            elif dataset == 'kitti-fine':
                bool_matrix = np.load('class_stats_fine_val.npy')
            elif dataset == 'kitti-finer':
                bool_matrix = np.load('class_stats_finer_val.npy')
            elif dataset == 'poss':
                bool_matrix = np.load('class_stats_poss_val.npy')
            elif dataset == 'nusc':
                bool_matrix = np.load('class_stats_nusc_val.npy')
            flag = True
    if flag:
        while(True):
            # include classes
            elems = input("type include classes: ").split(' ')
            in_classes = []
            for elem in elems:
                if len(elem) == 0:
                    break
                in_classes.append(int(elem))
            print(f'selected {len(in_classes)} include classes: {in_classes}')
            
            # exclude classes
            elems = input("type exclude classes: ").split(' ')
            # elems = input("type exclude classes: ").split(',')
            ex_classes = []
            for elem in elems:
                if len(elem) == 0:
                    break
                ex_classes.append(int(elem))
            print(f'selected {len(ex_classes)} exclude classes: {ex_classes}')
            
            mask_in = bool_matrix[:, in_classes].all(1)
            mask_ex = ~bool_matrix[:, ex_classes].any(1)
            mask = mask_in * mask_ex
            
            num_scan_possible = mask.sum()
            print(f'possible number of scan: {num_scan_possible}')
            
            yes = input("type y to save data idx: ")
            if yes == 'y':
                selected_data_idx = np.nonzero(mask)[0]
                if dataset == 'kitti':
                    if data_split == 'train':
                        np.save(f"selected_idx_in_{'_'.join([str(cls) for cls in sorted(in_classes)])}_ex_{'_'.join([str(cls) for cls in sorted(ex_classes)])}.npy", selected_data_idx)
                    elif data_split == 'valid':
                        np.save(f"selected_idx_valid_in_{'_'.join([str(cls) for cls in sorted(in_classes)])}_ex_{'_'.join([str(cls) for cls in sorted(ex_classes)])}.npy", selected_data_idx)
                elif dataset == 'nusc':
                    if data_split == 'train':
                        np.save(f"nusc_selected_idx_in_{'_'.join([str(cls) for cls in sorted(in_classes)])}_ex_{'_'.join([str(cls) for cls in sorted(ex_classes)])}.npy", selected_data_idx)
                    elif data_split == 'valid':
                        np.save(f"nusc_selected_idx_valid_in_{'_'.join([str(cls) for cls in sorted(in_classes)])}_ex_{'_'.join([str(cls) for cls in sorted(ex_classes)])}.npy", selected_data_idx)
                else:
                    raise NotImplementedError
    else:
        if dataset == 'kitti':
            dataset_config_path = 'config/semkitti_dataset.yaml'
        elif dataset == 'kitti-fine':
            dataset_config_path = 'config/semkitti_fine_dataset.yaml'
        elif dataset == 'kitti-finer':
            dataset_config_path = 'config/semkitti_finer_dataset.yaml'
        elif dataset == 'poss':
            dataset_config_path = 'config/semposs_dataset.yaml'
        elif dataset == 'nusc':
            dataset_config_path = 'config/nuscenes_dataset.yaml'
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        voxel_size = 0.05
        downsampling = 80000
        
        unknown_labels = []
        number_of_unk=len(unknown_labels)
        
        label_mapping, label_mapping_inv, unknown_label = unk_labels.label_mapping(
            unknown_labels, dataset_config['learning_map_inv'].keys())
        
        if (dataset == 'kitti') or (dataset == 'kitti-fine') or (dataset == 'kitti-finer'):
            Dataset = SemanticKITTIRestrictedDataset(
                    config_file=dataset_config_path,
                    split=data_split,
                    voxel_size=voxel_size,
                    downsampling=downsampling,
                    augment=False,
                    label_mapping=label_mapping,
                )
        elif dataset == 'poss':
            Dataset = SemanticPOSSRestrictedDataset(
                    config_file=dataset_config_path,
                    split=data_split,
                    voxel_size=voxel_size,
                    downsampling=downsampling,
                    augment=False,
                    label_mapping=label_mapping,
                )
        elif dataset == 'nusc':
            from nuscenes import NuScenes
            nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_config['dataset_path'], verbose=True)
            Dataset = nuScenesRestrictedDataset(
                    config_file=dataset_config_path,
                    split=data_split,
                    voxel_size=voxel_size,
                    downsampling=downsampling,
                    augment=False,
                    label_mapping=label_mapping,
                    nusc=nusc,
                )

        bool_matrix = np.zeros((len(Dataset), len(label_mapping))).astype(bool)
        
        for i in tqdm(range(len(Dataset))):
            # coordinates, features, mapped_labels, selected_idx, labels, t = dataset.get_raw_data(i)
            coordinates, features, selected_idx, labels, t = Dataset.get_raw_data(i)
            
            # if (20 in np.unique(labels)) or (21 in np.unique(labels)):
            #     breakpoint()
            # unq_labels = np.unique(labels)
            # bool_matrix[i, unq_labels] = True
            
            bool_matrix[i] = nb_function(bool_matrix[i], labels.astype(np.uint8))
        
        if data_split == 'train':
            if dataset == 'kitti':
                np.save('class_stats.npy', bool_matrix)
            elif dataset == 'kitti-fine':
                np.save('class_stats_fine.npy', bool_matrix)
            elif dataset == 'kitti-finer':
                np.save('class_stats_finer.npy', bool_matrix)
            elif dataset == 'poss':
                np.save('class_stats_poss.npy', bool_matrix)
            elif dataset == 'nusc':
                np.save('class_stats_nusc.npy', bool_matrix)
        elif data_split == 'valid':
            if dataset == 'kitti':
                np.save('class_stats_val.npy', bool_matrix)
            elif dataset == 'kitti-fine':
                np.save('class_stats_fine_val.npy', bool_matrix)
            elif dataset == 'kitti-finer':
                np.save('class_stats_finer_val.npy', bool_matrix)
            elif dataset == 'poss':
                np.save('class_stats_poss_val.npy', bool_matrix)
            elif dataset == 'nusc':
                np.save('class_stats_nusc_val.npy', bool_matrix)
        # np.save('class_stats_val.npy', bool_matrix)
    
    print('done')