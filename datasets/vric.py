import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class VRIC(BaseImageDataset):
    """
       Vehicle Re-Identificaition in Context (VRIC)
       Reference:
       Kanaci, A., Zhu, X., Gong, S.: Vehicle Re-Identificaition in Context
       German Conference on Patttern Recognition (2018)

       URL:https://qmul-vric.github.io/

       Dataset statistics:
        2811 train identities with 54808 images
        2811 test identities with 5622 images.
       """

    dataset_dir = 'VRIC'

    def __init__(self, root='', verbose=True, **kwargs):
        super(VRIC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_images')
        self.query_dir = osp.join(self.dataset_dir, 'probe_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_images')

        self._check_before_run()
        
        self.image_map= {}

        path_train = self.dataset_dir + "/vric_train.txt"
        with open(path_train, 'r') as txt:
            lines = txt.readlines()
        for img_idx, img_info in enumerate(lines):
            img_name, pid, camid = img_info.split(' ')
            self.image_map[osp.basename(img_name)] = {
                'pid': int(pid),
                'camid': int(camid),
            }

        path_query = self.dataset_dir + "/vric_probe.txt"
        with open(path_query, 'r') as txt:
            lines = txt.readlines()
        for img_idx, img_info in enumerate(lines):
            img_name, pid, camid = img_info.split(' ')
            self.image_map[osp.basename(img_name)] = {
                'pid': int(pid),
                'camid': int(camid),
            }

        path_test = self.dataset_dir + "/vric_gallery.txt"
        with open(path_test, 'r') as txt:
            lines = txt.readlines()
        for img_idx, img_info in enumerate(lines):
            img_name, pid, camid = img_info.split(' ')
            self.image_map[osp.basename(img_name)] = {
                'pid': int(pid),
                'camid': int(camid),
            }

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> VRIC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))[:10]

        if relabel:
            pid_container = set()
            for img_path in img_paths:
                img_name = osp.basename(img_path)
                pid = self.image_map[img_name]['pid']
                if pid == -1: continue  # junk images are just ignored
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

        camid_container = set()
        dataset = []
        count = 0
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            try:
                pid = self.image_map[img_name]['pid']
                camid = self.image_map[img_name]['camid']
            except:
                count += 1
                continue
            camid_container.add(camid)
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid, camid, 1))
        print(count, 'samples without annotations')
        print('camid container', camid_container)
        return dataset

