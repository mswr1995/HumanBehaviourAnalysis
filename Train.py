from VideoPreprocessing import VideoNameDF, TagVideos, Video2Frames, FramesCSV
from Model import ReadFrames, DatasetSplit, VGG16Model, UCFModel_Train


###################### Video Preprocessing ########################
# creating a dataframe having video names
train = VideoNameDF(name='trainlist01.txt', dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/ucfTrainTestlist')
test = VideoNameDF(name='testlist01.txt', dir='D:Classes/Thesis/HumanBehaviourAnalysis/UCF/ucfTrainTestlist')

#Next, we will add the tag of each video (for both training and test sets).
train = TagVideos(train)
test = TagVideos(test)

#Extracting frames from training videos.
path = 'D:/Classes/Thesis/HumanBehaviourAnalysis/UCF'
status = Video2Frames(train, frames_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/train_1', videos_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/UCF-101')
print(status)

#Save the names of the frames to a CSV file along with their corresponding tags.
train_data = FramesCSV(frames_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/train_1', csv_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF', csv_name='train_new.csv')

################### Data Preprocessing ##########################

import pandas as pd
train = pd.read_csv('D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/train_new.csv')
train.head()

X = ReadFrames(train, frames_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/train_1')
#Creating the test set and validation set.
y = train['class']
X_train, X_test, y_train, y_test = DatasetSplit(X, y)
#delete the variable X to save the space.
del X
################### Train the Model ##########################
X_train, X_test = VGG16Model(X_train, X_test)
history = UCFModel_Train(X_train, y_train, X_test, y_test, epochs=500, ckpt_name='UCF_weights.h5')



