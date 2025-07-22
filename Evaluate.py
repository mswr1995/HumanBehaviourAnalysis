import pandas as pd
from keras.applications.vgg16 import VGG16
from Model import UCFModel, PredictAction, PredictOnBatch
from VideoPreprocessing import VideoNameDF
from sklearn.metrics import accuracy_score
from keras.models import load_model


# path = 'D:/Classes/Thesis/HumanBehaviourAnalysis/UCF'

base_model = VGG16(weights='imagenet', include_top=False)
model = UCFModel(shape=25088)
# loading the trained weights
model.load_weights('D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/ckpt/UCF_weights.h5')
# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# getting the test list
test = VideoNameDF(name='testlist01.txt', dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/ucfTrainTestlist')
test_videos = test['video_name']

# creating the tags
train = pd.read_csv('D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/train_new.csv')
y = train['class']
y = pd.get_dummies(y)


################# Making a prediction on single video ################
file='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/v_Archery_g03_c03.avi'
action = PredictAction(file, y, model, base_model)
print(action)
    
################# Making a prediction on batch of videos ################
predict, actual = PredictOnBatch(test_videos[0:100], y, base_model, model, videos_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF/UCF-101')
# checking the accuracy of the predicted tags
acc = accuracy_score(predict, actual)*100
print('\n'+str(acc))


