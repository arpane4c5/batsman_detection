#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 02:47:52 2019

@author: arpan

@Description: Extract frames in the dataset and save to disk
"""

import cv2
import os
import time
import utils
    

def extract_vid_frames(srcFolderPath, destFolderPath, vfiles, stop='all'):
    """
    Function to extract the features from a list of videos
    
    Parameters:
    ------
    srcFolderPath: str
        path to folder which contains the videos
    destFolderPath: str
        path to store the frame pixel values in .npy files
    vfiles: list 
        list of filenames (without ext)
    stop: str
        to traversel 'stop' no of files in each subdirectory.
    
    Returns: 
    ------
    traversed: int
        no of videos traversed successfully
    """
    
    traversed = 0
    # create destination path to store the files
    if not os.path.exists(destFolderPath):
        os.makedirs(destFolderPath)
        
    # iterate over the video files inside the directory
    for in_file in vfiles:
        if os.path.isfile(os.path.join(srcFolderPath, in_file+".avi")):
            in_file = in_file+".avi"
        elif os.path.isfile(os.path.join(srcFolderPath, in_file+".mp4")):
            in_file = in_file + ".mp4"
        else:
            print("Check file extension for "+in_file)
            break
            
            # save at the destination, if extracted successfully
        written = writeFrames(srcFolderPath, in_file, destFolderPath)
        traversed += 1
        print("Done "+str(traversed)+" :: "+in_file)
                    
        # to stop after successful traversal of 2 videos, if stop != 'all'
        if stop != 'all' and traversed == stop:
            break
                    
    print("No. of files written to destination : "+str(traversed))
    if traversed == 0:
        print("Check the structure of the dataset folders !!")
    ###########################################################################
    return traversed


def writeFrames(srcFolderPath, srcVideo, destFolder):
    """
    Function to write all the frames of the video file into the destination folder.
    """
    # get the VideoCapture object
    cap = cv2.VideoCapture(os.path.join(srcFolderPath, srcVideo))
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
#    W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    
    #ret, prev_frame = cap.read()
    assert cap.isOpened(), "Capture object does not return a frame!"
    
    vid_prefix = srcVideo.rsplit('.', 1)[0]
    # Iterate over the entire video to get the optical flow features.
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imwrite(os.path.join(destFolder, \
                        vid_prefix+"_{:012}".format(frameCount)+'.png'), frame)
        #direction = waitTillEscPressed()
        frameCount +=1

    # When everything done, release the capture
    #cv2.destroyAllWindows()
    cap.release()
    return True


if __name__=='__main__':
    
    srcPath = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    destPath = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/ICC_WT20_frames"
    if not os.path.exists(srcPath):
        srcPath = "/opt/datasets/cricket/ICC_WT20"
        destPath = "/home/arpan/DATA_Drive/Cricket/Workspace/batsman_pose_track/ICC_WT20_frames"
    
    train_lst, val_lst, test_lst = utils.split_dataset_files(srcPath)
    
    start = time.time()
    nfiles = extract_vid_frames(srcPath, destPath+"/train", train_lst, stop='all')
    nfiles = extract_vid_frames(srcPath, destPath+"/val", val_lst, stop='all')
    nfiles = extract_vid_frames(srcPath, destPath+"/test", test_lst, stop='all')
    end = time.time()
    print("Total execution time for {} files : {}".format(nfiles, str(end-start)))
    
