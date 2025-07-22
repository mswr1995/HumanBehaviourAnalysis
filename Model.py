from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout
from keras.utils import load_img, img_to_array
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from VideoPreprocessing import FrameExtractor, LoadImages
from scipy import stats as s

def ReadFrames(train, frames_dir):
    """  read the frames that we extracted earlier and then store those frames as a NumPy array.
        Returns a Numpy array."""
    # creating an empty list
    train_image = []
    # for loop to read and store frames
    for i in tqdm(range(train.shape[0])):
        # loading the image and keeping the target size as (224,224,3)
        img = load_img(frames_dir+'/'+train['image'][i], target_size=(224, 224, 3))
        # converting it to array
        img = img_to_array(img, dtype=np.float16)
        # normalizing the pixel value
        img = img/255
        # appending the image to the train_image list
        train_image.append(img)    
    # with open("images", "wb") as fp:
    #     pickle.dump(prediction_images, fp)
    # converting the list to numpy array
    X = np.array(train_image)
    # shape of the array
    X.shape
    return X

def DatasetSplit(X, y):
    #Creating the test set and validation set.
    # separating the target
    """ To create the validation set, we need to make sure that the distribution of each class is similar 
        in both training and validation sets. stratify = y (which is the class or tags of each frame) keeps 
        the similar distribution of classes in both the training as well as the validation set."""
    
    # creating the training and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)
    
    # creating dummies of target variable for train and validation set
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    return X_train, X_test, y_train, y_test 

def VGG16Model(X_train, X_test):
    """ Intialize the Pre-trained VGG16 model, extract the features for given input, flatten
        the input, normalize them and then return a Numpy array."""
    # creating the base model of pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    # extracting features for training frames
    X_train = base_model.predict(X_train)
    print(X_train.shape)
    # extracting features for validation frames
    X_test = base_model.predict(X_test)
    print(X_test.shape)
    # reshaping the training as well as validation frames in single dimension
    X_train = X_train.reshape(round(len(X_train)), 7*7*512)
    X_test = X_test.reshape(round(len(X_test)), 7*7*512)
    # normalizing the pixel values
    max = X_train.max()
    X_train = X_train/max
    X_test = X_test/max
    print(X_train.shape)
    return X_train, X_test
    
def UCFModel(shape):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(101, activation='softmax'))
    return model
    
def UCFModel_Train(X_train, y_train, X_test, y_test, epochs, ckpt_name='UCF_weights.h5'):
    #defining the model architecture
    model = UCFModel(shape=25088)
    # defining a function to save the weights of best model
    ckpt = ModelCheckpoint(ckpt_name, save_best_only=True, monitor='val_loss', mode='min')
    # compiling the model
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    #priniting summary
    model.summary()
    # training the model
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                        callbacks=[ckpt], batch_size=128)
    return history

def PredictAction(file, y, model, base_model):
    name = file.split('/')[1]
    _ = FrameExtractor(name, frames_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF', videos_dir=file.split('/')[0])
    prediction_images = LoadImages(frames_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF')
    # extracting features using pre-trained model
    prediction_images = base_model.predict(prediction_images)
    # converting features in one dimensional array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
    # predicting tags for each array
    prediction = model.predict_classes(prediction_images)
    prediction = y.columns.values[s.mode(prediction)[0][0]]
    return prediction

def PredictOnBatch(test_videos, y, base_model, model, videos_dir):
    # creating two lists to store predicted and actual tags
    predict = []
    actual = []
    # for loop to extract frames from each test video
    for i in tqdm(range(test_videos.shape[0])):
        videoFile = FrameExtractor(test_videos, index=i, frames_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF', videos_dir=videos_dir)
        prediction_images = LoadImages(frames_dir='D:/Classes/Thesis/HumanBehaviourAnalysis/UCF')
        # extracting features using pre-trained model
        prediction_images = base_model.predict(prediction_images)
        # converting features in one dimensional array
        prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
        # predicting tags for each array
        prediction = model.predict_classes(prediction_images)
        # appending the mode of predictions in predict list to assign the tag to the video
        predict.append(y.columns.values[s.mode(prediction)[0][0]])
        # appending the actual tag of the video
        vid = videoFile.split('/')
        vid = vid[len(vid)-1]
        actual.append(vid.split('_')[1])
    return predict, actual