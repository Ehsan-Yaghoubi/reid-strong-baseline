#
"""
@author:  Ehsan Yaghoubi
@contact: Ehsan.yaghoubi@gmail.com
"""

import glob
import re
import os
from .bases import BaseImageDataset
import mat4py

def load_reid_data(rap_mat_file):
    """
    For each person in RAP dataset, we have sampled one image from one camera at one day as the query,
    so one person may have several queries from one camera in different days. To change this setting,
    we refer you to https://github.com/dangweili/RAP/blob/master/person-reid/evaluation/rap2_evaluation_features.m
    """
    RAP_dictionary = mat4py.loadmat(rap_mat_file)
    name_of_imgs = RAP_dictionary["RAP_annotation"]["name"]
    all_person_ids = RAP_dictionary["RAP_annotation"]["person_identity"][:41585]
    train_ids_indices = RAP_dictionary["RAP_annotation"]["partition_reid"]["train_index"]
    gallery_ids_indices = RAP_dictionary["RAP_annotation"]["partition_reid"]["gallery_index"]
    query_ids_indices = RAP_dictionary["RAP_annotation"]["partition_reid"]["query_index"]

    name_of_imgs = [subsub for sub in name_of_imgs for subsub in sub]
    all_person_ids = [subsub for sub in all_person_ids for subsub in sub]
    train_ids_indices = [subsub for sub in train_ids_indices for subsub in sub]
    gallery_ids_indices = [subsub for sub in gallery_ids_indices for subsub in sub]
    #query_ids_indices = [subsub for sub in query_ids_indices for subsub in sub]

    return name_of_imgs, all_person_ids, train_ids_indices, gallery_ids_indices, query_ids_indices

class RAP (BaseImageDataset):

    dataset_dir = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/RAP_resized_imgs'
    rap_mat_file = '/media/ehsan/48BE4782BE476810/AA_GITHUP/Anchor_Level_Paper/rap_annotations/RAP_annotation.mat'

    def __init__(self, verbose=True, **kwargs):
        super(RAP, self).__init__()
        self._check_before_run()

        train, query, gallery = self._process_dir(self.rap_mat_file, self.dataset_dir, relabel= True)

        self.train = train
        self.query = query
        self.gallery = gallery


        if verbose:
            print("=> RAP is loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        #assert self.num_train_pids == 1295


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.rap_mat_file):
            raise RuntimeError("'{}' is not available".format(self.rap_mat_file))


    def _process_dir(self, rap_mat_file, data_dir, relabel):
        image_names, pids, train_indices, gallery_indices, query_indices=load_reid_data(rap_mat_file)
        pid_container = set()
        for index, IMG_index1 in enumerate(train_indices):
            person_id = pids[IMG_index1-1]
            if person_id == -1: continue  # junk images are just ignored
            pid_container.add(person_id)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        train = []
        ids = []
        cam_ids = []
        for index, IMG_index1 in enumerate(train_indices):
                person_id = pids[IMG_index1-1]
                if person_id == -1: continue  # junk images are just ignored
                img_name = image_names[IMG_index1-1]
                img_full_path = os.path.join(data_dir, img_name)
                camera = img_name.split("-")[0] # e.g. ['CAM01', '2014', '02', '15', '20140215161032', '20140215162620', 'tarid0', 'frame218', 'line1.png']
                camera_id = int(re.findall('\d+',camera)[0])
                assert 1 <= person_id <= 2589  # There are 2589 person identities in RAP dataset
                assert 1 <= camera_id <= 31 # There are 23 cameras with labels between 1 to 31
                cam_ids.append(camera_id)
                ids.append(person_id)
                if relabel: person_id = pid2label[person_id] # train ids must be relabelled from zero
                train.append((img_full_path, person_id , camera_id))

        print(">> Number of train Images: ", len(set(train)))
        print(">> Number of train IDs: ",len(set(ids)))
        print(">> Number of train Camera IDs: ",len(set(cam_ids)))

        query= []
        ids = []
        cam_ids = []
        for i2, IMG_index2 in enumerate(query_indices):
                person_id = pids[IMG_index2-1]
                if person_id == -1: continue  # junk images are just ignored
                img_name = image_names[IMG_index2-1]
                img_full_path = os.path.join(data_dir, img_name)
                camera = img_name.split("-")[0] # e.g. ['CAM01', '2014', '02', '15', '20140215161032', '20140215162620', 'tarid0', 'frame218', 'line1.png']
                camera_id = int(re.findall('\d+',camera)[0])
                assert 1 <= person_id <= 2589  # There are 2589 person identities in RAP dataset
                assert 1 <= camera_id <= 31 # There are 23 cameras with labels between 1 to 31
                cam_ids.append(camera_id)
                ids.append(person_id)
                query.append((img_full_path, person_id , camera_id))

        print(">>>> Number of query Images: ",len(set(query)))
        print(">>>> Number of query IDs: ",len(set(ids)))
        print(">>>> Number of query Camera IDs: ",len(set(cam_ids)))

        gallery= []
        ids = []
        cam_ids = []
        for i3, IMG_index3 in enumerate(gallery_indices):
                person_id = pids[IMG_index3-1]
                if person_id == -1: continue  # junk images are just ignored
                img_name = image_names[IMG_index3-1]
                img_full_path = os.path.join(data_dir, img_name)
                camera = img_name.split("-")[0] # e.g. ['CAM01', '2014', '02', '15', '20140215161032', '20140215162620', 'tarid0', 'frame218', 'line1.png']
                camera_id = int(re.findall('\d+',camera)[0])
                assert 1 <= person_id <= 2589  # There are 2589 person identities in RAP dataset
                assert 1 <= camera_id <= 31 # There are 23 cameras with labels between 1 to 31
                cam_ids.append(camera_id)
                ids.append(person_id)
                gallery.append((img_full_path, person_id , camera_id))

        print(">>>>>> Number of gallery Images: ",len(set(gallery)))
        print(">>>>>> Number of gallery IDs: ",len(set(ids)))
        print(">>>>>> Number of gallery Camera IDs: ",len(set(cam_ids)))

        return  train, query, gallery

