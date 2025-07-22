# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:43:03 2020

@author: danish
"""

import cv2     # for capturing videos
import math   # for mathematical operations
import pandas as pd
from keras.utils import load_img, img_to_array
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import shutil

def VideoNameDF(name, dir):
    """ We will now store the name of videos in a dataframe.
        Returns a datframe. Containing names of videos."""
    # open the .txt file which have names of videos
    f = open('D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/ucfTrainTestlist/testlist01.txt', "r")
    temp = f.read()
    # Spliting the videos by new line.
    videos = temp.split('\n')
    # creating a dataframe having video names
    train = pd.DataFrame()
    train['video_name'] = videos
    train = train[:-1]
    train.head()
    return train

def TagVideos(df):
    """ The entire part before the ‘/’ in the video name represents the tag of the video,
        Hence, we will split the entire string on ‘/’ and select the tag for all the videos.
        Returns a datframe"""
    # creating tags for videos
    video_tag = []
    for i in range(df.shape[0]):
        video_tag.append(df['video_name'][i].split('/')[0])
    df['tag'] = video_tag
    return df


def Video2Frames(df, frames_dir='train_1', videos_dir='/UCF'):
    # storing the frames from training videos
    """  Extract the frames from the training videos which will be used to train the model.
        Store the frames in the given frames directory."""
    os.makedirs(frames_dir, exist_ok=True)
    for i in tqdm(range(df.shape[0])):
        count = 0
        videoFile = df['video_name'][i]
        # capturing the video from the given path
        # cap = cv2.VideoCapture(videos_dir+'/'+videoFile.split(' ')[0].split('/')[0])
        cap = cv2.VideoCapture(videos_dir+'/'+videoFile.split(' ')[0])
        frameRate = cap.get(5) #frame rate
        # x=1
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                # storing the frames in a new folder named train_1
                filename = frames_dir + '/' + videoFile.split('/')[1].split(' ')[0] +"_frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        cap.release()
    return '\nFrames are extracted from the videos succesfully!'


def FramesCSV(frames_dir, csv_dir='UCF', csv_name='train_new.csv'):
    """ Save the name of the frames with their corresponding tag in a .csv file. 
        Creating this file will help us to read the frames.
        Returns a datframe containing the name of all the frames."""
    # getting the names of all the images
    images = glob("{0}/*.jpg".format(frames_dir))
    images = [path.replace('\\', '/') for path in images]
    train_image = []
    train_class = []
    for i in tqdm(range(len(images))):
        # creating the image name
        #Following line will extract the last part of the path which is image name.
        index = len(images[i].split('/'))-1
        train_image.append(images[i].split('/')[index])
        # creating the class of image
        train_class.append(images[i].split('/')[index].split('_')[1])
    # storing the images and their class in a dataframe
    train_data = pd.DataFrame()
    train_data['image'] = train_image
    train_data['class'] = train_class
    # converting the dataframe into csv file 
    os.makedirs(csv_dir, exist_ok=True)
    train_data.to_csv('{0}/{1}'.format(csv_dir, csv_name),header=True, index=False)
    return train_data

def FrameExtractor(test_videos, index=None, frames_dir='D:/Classes\Thesis/HumanBehaviourAnalysis/UCF/train_1', videos_dir='UCF-101'):
    """ Extract the frames from given video clip and store in the given directory."""
    if type(test_videos) == str:
        test_videos = [test_videos]
        index=0
    if index==None:
        raise ValueError('Invalid value for `argument` index.')
    count = 0
    #Setting the permission and removing `frames_dir` directory which may contain old files.
    #os.chmod(frames_dir, 0o777)
    shutil.rmtree(frames_dir, ignore_errors=True)
    #Creating the new directory
    os.makedirs(frames_dir, exist_ok=True)
    videoFile = test_videos[index]
    # capturing the video from the given path
    cap = cv2.VideoCapture(videos_dir+'/'+videoFile.split(' ')[0])
    frameRate = cap.get(5) #frame rate
    # removing all other files from the temp folder
    # files = glob(frames_dir)
    # for f in files:
    #     os.remove(f)
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        #print(frameId)
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames of this particular video in temp folder
            filename =frames_dir+'/' + "_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()
    return videoFile

def LoadImages(frames_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/train_1'):
    """ Read the frames from the given directory process them and make list of all the frames
        which belongs to a video, convert the list into Numpy array and return that array."""
    # reading all the frames from temp folder
    images = glob(frames_dir+"/*.jpg")
    prediction_images = []
    for i in range(len(images)):
        img = load_img(images[i], target_size=(224,224,3))
        img = img_to_array(img)
        img = img/255
        prediction_images.append(img)
    # converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
    return prediction_images
