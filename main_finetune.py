#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 02:17:27 2019

@author: arpan
"""

#Writing custom dataset for Batsman Detection

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from collections import Counter
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#import math


TRAIN_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/train"
VAL_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/val"
TEST_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/test"
ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
BASE_PATH = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/logs"
# let's train it for 10 epochs
num_epochs = 4

class BatsmanDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, gt_path, transforms=None):
        self.root = root
        self.transforms = transforms
        self.gt_path = gt_path
        # read all files and find unique video names
        all_files = os.listdir(root)
        #all_files_set = list(set([f.rsplit("_", 1)[0] for f in all_files]))  #unique video prefixes
        # get number of frames in each video in dictionary
        all_files_dict = dict(Counter([f.rsplit("_", 1)[0] for f in  all_files]))   
        #print(all_files_dict)
        self.img_paths = [key+"_{:012}".format(i)+".png" for key in \
                          sorted(list(all_files_dict.keys())) \
                          for i in range(all_files_dict[key])]
        self.bboxes = self.get_annotation_boxes(all_files_dict)
        
        self.bboxes_pos = []
        self.img_paths_pos = []
        for idx, box in enumerate(self.bboxes):
            if box!=[]:
                self.bboxes_pos.append(box)
                self.img_paths_pos.append(self.img_paths[idx])
                
#        self.bboxes = self.bboxes[:1000]
#        self.img_paths_pos = self.img_paths_pos[:1000]
        
        
    def get_annotation_boxes(self, keys_dict):
        ''' Create list of boxes for all the frames in the dataset.
        '''
        boxes = []
        # Iterate the video frames in the same order as done for img_paths
        for key in sorted(list(keys_dict.keys())):
            vid_nFrames = keys_dict[key]
            
            with open(os.path.join(self.gt_path, key+"_gt.txt"), "r") as fp:
                f = fp.readlines()
            
            # # remove \n at end and split into list of tuples
            # eg. tuple is ['98', '1', '303', '28', '353', '130', 'Batsman']
            f = [line.strip().split(',') for line in f]   
            f.reverse()
            frame_label = None
            
            for i in range(vid_nFrames):
                if frame_label == None:
                    if len(f) > 0:
                        frame_label = f.pop()
                    
                if frame_label is not None and int(frame_label[0])==i and \
                    int(frame_label[1])==1 and frame_label[-1]=='Batsman':
                    xmin = int(frame_label[2])
                    ymin = int(frame_label[3])
                    xmax = int(frame_label[4])
                    ymax = int(frame_label[5])
                    boxes.append([xmin, ymin, xmax, ymax])
                    frame_label = None
                else:
                    boxes.append([])
                    
        return boxes
        
        
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.img_paths_pos[idx])
        img = Image.open(img_path).convert("RGB")

        fr = img_path.rsplit(".", 1)[0].rsplit("_", 1)[1]
        fr_id = int(fr)
        
        box = self.bboxes_pos[idx]
        boxes = []
        num_objs = 1   # for only Batsman
        #if box!=[]:
        #    boxes.append(box)
        #    num_objs = 0
        boxes.append(box)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        frame_id = torch.tensor([fr_id])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["frame_id"] = frame_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths_pos)


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def save_model_checkpoint(base_name, model, ep, opt):
    """
    TODO: save the optimizer state with epoch no. along with the weights
    """
    if not os.path.exists(base_name):
        os.makedirs(base_name)
    # Save only the model params
    name = os.path.join(base_name, "FasterRCNN_resnet50_ep"+str(ep)+"_"+opt+".pt")
#    if use_gpu and torch.cuda.device_count() > 1:
#        model = model.module    # good idea to unwrap from DataParallel and save

    torch.save(model.state_dict(), name)
    print("Model saved to disk... {}".format(name))
    

if __name__ == '__main__':
    
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    
    # use our dataset and defined transformations
    dataset = BatsmanDetectionDataset(TRAIN_FRAMES, ANNOTATION_FILE, \
                                      get_transform(train=True))
    dataset_test = BatsmanDetectionDataset(TEST_FRAMES, ANNOTATION_FILE, \
                                           get_transform(train=False))
    
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:])
    indices_test = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test[:])
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, \
                                              num_workers=4, collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, \
                                shuffle=False, num_workers=4, collate_fn=utils.collate_fn)
    
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    lrate = 0.005
    optimizer = torch.optim.SGD(params, lr=lrate, momentum=0.9, weight_decay=0.0005)
    
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
#    mod_file = os.path.join(BASE_PATH, \
#                    "FasterRCNN_resnet50_ep"+str(num_epochs)+"_SGD.pt")
#    if os.path.isfile(mod_file):
#        model.load_state_dict(torch.load(mod_file))
#    
#    for epoch in range(num_epochs):
#        # train for one epoch, printing every 10 iterations
#        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=200)
#        # update the learning rate
#        lr_scheduler.step()
#        # evaluate on the test dataset
#        evaluate(model, data_loader_test, device=device)
#        
#        if (epoch+1)%2 == 0:
#            save_model_checkpoint(BASE_PATH, model, epoch+1, "SGD")
            
    mod_file = os.path.join(BASE_PATH, \
                    "FasterRCNN_resnet50_ep"+str(num_epochs)+"_SGD.pt")
    model.load_state_dict(torch.load(mod_file))
    evaluate(model, data_loader_test, device=device)
