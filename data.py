#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import open3d as o3d

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def subsample_points_low(pointcloud1, pointcloud2, num_subsampled_points, rotation_ab, translation_ab):
    # (num_points, 3)
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_idx1 = np.random.choice(num_points)
    random_p1 = pointcloud1[random_idx1, :]
    distance = np.sum((pointcloud2 - random_p1) ** 2, axis=-1)
    random_idx2 = np.argmax(distance)
    random_p2 = pointcloud1[random_idx2, :]
    idx1 = nbrs1.kneighbors(random_p1.reshape(1, -1), return_distance=False).reshape((num_subsampled_points,))
    idx2 = nbrs2.kneighbors(random_p2.reshape(1, -1), return_distance=False).reshape((num_subsampled_points,))
    pointcloud1 = pointcloud1[idx1, :]
    pointcloud2 = pointcloud2[idx2, :]
    pointcloud2 = rotation_ab.apply(pointcloud2).T + np.expand_dims(translation_ab, axis=1)
    return pointcloud1.T, pointcloud2

def subsample_points_moderate(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T

def subsample_points_large(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))

    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T
def view_points(xyz_points1, xyz_points2):
    # pcd1 = o3d.geometry.PointCloud()
    color1 = np.array([255, 0, 0], dtype='uint8').reshape(1, -1)
    color1 = np.repeat(color1, xyz_points1.shape[0], axis=0)
    color2 = np.array([0, 0, 255], dtype='uint8').reshape(1, -1)
    color2 = np.repeat(color2, xyz_points2.shape[0], axis=0)

    points1 = o3d.geometry.PointCloud()
    points1.points = o3d.utility.Vector3dVector(xyz_points1)
    points1.colors = o3d.utility.Vector3dVector(color1)

    points2 = o3d.geometry.PointCloud()
    points2.points = o3d.utility.Vector3dVector(xyz_points2)
    points2.colors = o3d.utility.Vector3dVector(color2)
    o3d.visualization.draw_geometries([points1, points2])

class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train',
                 gaussian_noise=False, unseen=False, rot_factor=4, category=None, overlap='low'):
        super(ModelNet40, self).__init__()
        self.data, self.label = load_data(partition)
        if category is not None:
            self.data = self.data[self.label==category]
            self.label = self.label[self.label==category]
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.overlap = overlap
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.rot_factor = rot_factor
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]
        self.no_exact_corr = True
    def __getitem__(self, item):
        pointcloud1 = self.data[item][:self.num_points]
        if self.no_exact_corr is True:
            pointcloud2 = self.data[item][self.num_points:].T
        else:
            pointcloud2 = pointcloud1.T
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud1.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]
        if self.subsampled:
            if self.overlap == 'low':
                pointcloud1, pointcloud2 = subsample_points_low(pointcloud1, pointcloud2,
                                                                self.num_subsampled_points, rotation_ab, translation_ab)
            elif self.overlap == 'moderate':
                pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
                pointcloud1, pointcloud2 = subsample_points_moderate(pointcloud1, pointcloud2,
                                                                     num_subsampled_points=self.num_subsampled_points)
            else:
                pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
                pointcloud1, pointcloud2 = subsample_points_large(pointcloud1, pointcloud2,
                                                                       num_subsampled_points=self.num_subsampled_points)
        else:
            pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T
        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)
        # view_points(pointcloud1.T, pointcloud2.T)
        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    print('hello world')