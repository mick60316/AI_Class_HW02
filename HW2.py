import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from keras import optimizers
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import csv



Train_fpath='data/train.csv'
Exam_fpath='data/test.csv'
OutputFileName = "ex.csv"
threshold=0.5
Batchsize =500
Epochs =1
lr =0.0001
FeatureCount = 13

def ExamDataPreprocessing (ExamData):
    ExamData['workclass']=ExamData['workclass'].replace(' ?',' Private')
    ExamData['workclass'].fillna('Private')
    ExamData['workclass'].replace(np.nan, 'Private', inplace=True)
    ExamData['workclass'].replace(np.inf, 'Private', inplace=True)
    ExamData['marital_status'].fillna(0)
    ExamData['marital_status'].replace(np.nan, '0', inplace=True)
    ExamData['marital_status'].replace(np.inf, '0', inplace=True)
    
    colnums =pd.Series(['workclass','education','marital_status','occupation','relationship','race','sex','native_country'])
    ExamData =DataAllDecode(ExamData,colnums)


    #ExamData.drop(['fnlwgt'],axis =1)
   

    ndarray =ExamData.values

    Features=ndarray[:,0:FeatureCount]
    
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)
    print(scaledFeatures)
    return scaledFeatures


def ReadFile (TrainPath,TestPath):
    Train =pd.read_csv (TrainPath)
    Test =pd.read_csv (TestPath)
    return Train,Test

def DataDecode (Data):
    Items =Data.drop_duplicates()
    index = 0
    for item in Items:
        Data.replace(item, index, inplace=True)
        index=index+1
    Data=Data.astype(int)
    return Data

def DataAllDecode (Data,colums):
    for index in colums:
        Data[index]=DataDecode(Data[index])
    return Data
def TrainDataPreprocessing(Data):

    Data['workclass']=Data['workclass'].replace(' ?',' Private')
    Data['workclass'].fillna('Private')
    Data['workclass'].replace(np.nan, 'Private', inplace=True)
    Data['workclass'].replace(np.inf, 'Private', inplace=True)
    Data['marital_status'].fillna(0)
    Data['marital_status'].replace(np.nan, '0', inplace=True)
    Data['marital_status'].replace(np.inf, '0', inplace=True)
    colnums =pd.Series(['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income'])
    Data =DataAllDecode(Data,colnums)
 
    
    #Data.to_csv("CleanData.csv")
    #Data.drop(['fnlwgt'],axis =1)
    x_OneHot_df = pd.get_dummies(data=Data,columns=["income"])
   

    ndarray =x_OneHot_df.values
    Label =ndarray[:,14]    
    Features=ndarray[:,0:FeatureCount]
    
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)
    print(scaledFeatures)
    return scaledFeatures,Label

   
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

def WriteOutputFile (AllProbability):
    
    with open(OutputFileName, 'w', newline='') as csvfile:
      
        writer = csv.writer(csvfile)
    
      
        writer.writerow(['id', 'label'])
        for i in range (0,len(AllProbability)):
            code= 1
            if(AllProbability[i] >threshold):
                code=0
            writer.writerow([i+1, code])
        print("Output file",OutputFileName)
    
if __name__=='__main__':
 
    
    TrainData ,ExamData =ReadFile(Train_fpath,Exam_fpath)
  
    ExamDataFeatures =ExamDataPreprocessing(ExamData)
    msk =np.random.rand(len(TrainData))<0.8
    train_data = TrainData[msk]
    test_data = TrainData[~msk]
    TrainFeatrue,TrainLabel = TrainDataPreprocessing(train_data)
    TestFeature,TestLabel =TrainDataPreprocessing(test_data)
    print(ExamDataFeatures[:2])
    model=Sequential()
    model.add(Dense(units =30,input_dim=FeatureCount,kernel_initializer='uniform',activation ='relu'))
    model.add(Dense(units =20,kernel_initializer='uniform',activation ='relu'))
    model.add(Dense(units =15,kernel_initializer='uniform',activation ='relu'))
    model.add(Dense(units =1,kernel_initializer='uniform',activation ='sigmoid'))
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    adam=optimizers.Adam (lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile (loss='binary_crossentropy',
                   optimizer=adam,metrics =['accuracy'])

    
    train_history =model.fit (x=TrainFeatrue,y=TrainLabel,
                              validation_split=0.2,epochs=Epochs,
                              batch_size =Batchsize,verbose =2)

    show_train_history(train_history,'acc','val_acc')
    show_train_history(train_history,'loss','val_loss')
    scores =model.evaluate (x =TestFeature,y =TestLabel)
    print(scores[1])

    all_probability=model.predict (ExamDataFeatures)
    
    print(all_probability,len(all_probability))
    
    WriteOutputFile(all_probability)
    


    
    
