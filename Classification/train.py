# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import cross_val_score,cross_validate

k = 10
import os
model_dir = "./" #"./models"


carb = "BWZ Carb Input (grams)"
sensor = "Sensor Glucose (mg/dL)"

inF1 = './InsulinData.csv'
inF2 = './Insulin_patient2.csv'
cgmF1 = './CGMData.csv'
cgmF2 = './CGM_patient2.csv'
k = 10


"""#### Meal Data"""

def mealDataExtraction(InsulinFile, CGMFile):

  InsulinData = pd.read_csv(InsulinFile)
  InsulinData = InsulinData[['Date', 'Time', carb ]]
  InsulinData['datetimestamp'] = pd.to_datetime(InsulinData['Date'] +" "+ InsulinData['Time'])
  InsulinData.drop(['Date',"Time"], axis=1, inplace=True)
  InsulinData.head()

  meal_filtered = InsulinData[InsulinData[carb].notnull()]
  meal_filtered=meal_filtered[meal_filtered[carb]>0]
  meal_filtered['diff'] = meal_filtered['datetimestamp'][::-1].diff()
  meal_filtered=meal_filtered[::-1]
  meal_filtered['hd']=meal_filtered['diff'] >=  pd.to_timedelta('0 days 02:00:00.00000')
  meal_filtered['flag'] = meal_filtered['hd'].shift(-1)
  meal_filtered['flag'] = meal_filtered['flag'].fillna(True)
  meal_filtered = meal_filtered[meal_filtered['flag']]


  values = [meal_filtered.iloc[i]['datetimestamp'] for i in range(len(meal_filtered['datetimestamp']))]
  
  CGMData = pd.read_csv(CGMFile)
  CGMData['datetimestamp'] = pd.to_datetime(CGMData['Date'] +" "+ CGMData['Time'])
  CGMData = CGMData[['Date', 'Time', 'datetimestamp',sensor]]


  mealData = []
  for val in values:
    start = val-pd.to_timedelta('0 days 00:30:00.00000')
    end = val+pd.to_timedelta('0 days 02:00:00.00000')
    tempData=CGMData[(CGMData['datetimestamp']>=start) & (CGMData['datetimestamp']<=end)]
    # mealData.append(tempData)
    if len(tempData[tempData['Sensor Glucose (mg/dL)'].isna()]) <= 6 and len(tempData)==30:
      mealData.append(tempData[::-1])

  return values, mealData

mealDates1, mealData1 = mealDataExtraction(inF1, cgmF1)
mealDates2, mealData2 = mealDataExtraction(inF2, cgmF2)
mealData = mealData1+mealData2


def nonMealDataExtraction(mealDates, InsulinFile, CGMFile):

  InsulinData = pd.read_csv(InsulinFile)
  InsulinData = InsulinData[['Date', 'Time', carb ]]
  InsulinData['datetimestamp'] = pd.to_datetime(InsulinData['Date'] +" "+ InsulinData['Time'])
  InsulinData.drop(['Date',"Time"], axis=1, inplace=True)
  InsulinData.head()

  no_meal_dates = []
  meal_values = mealDates
  i, n = 0, len(meal_values)
  lastInsulineDate = InsulinData.iloc[0]['datetimestamp']

  while i<n-1:
    noMealStart = meal_values[i]+pd.to_timedelta('0 days 02:00:00.00000')
    nmStart = InsulinData[InsulinData['datetimestamp']>noMealStart].iloc[-1]['datetimestamp']
    nmEnd = InsulinData[InsulinData['datetimestamp']>=(nmStart+pd.to_timedelta('0 days 02:00:00.00000'))].iloc[-1]['datetimestamp']
    val = InsulinData[(InsulinData['datetimestamp']>=nmStart) & (InsulinData['datetimestamp']<=nmEnd)]
    
    # print(val[carb].nunique())
    while len(val)>0 and not any(val[carb]>0.0) and nmEnd<meal_values[i+1]:
      no_meal_dates.append(nmStart)

      # if any(val[carb]>0.0):
      #   break
      nmStart = InsulinData[InsulinData['datetimestamp']>nmEnd].iloc[-1]['datetimestamp']
      nmEnd = InsulinData[InsulinData['datetimestamp']>=(nmStart+pd.to_timedelta('0 days 02:00:00.00000'))].iloc[-1]['datetimestamp']
      val = InsulinData[(InsulinData['datetimestamp']>=nmStart) & (InsulinData['datetimestamp']<=nmEnd)]
      # print(val[carb].nunique())

    i+=1
    
  else:

    if i == n-1:
      noMealStart = meal_values[i]+pd.to_timedelta('0 days 02:00:00.00000')
      nmStart = InsulinData[InsulinData['datetimestamp']>noMealStart].iloc[-1]['datetimestamp']
      nmEnd = InsulinData[InsulinData['datetimestamp']>=(nmStart+pd.to_timedelta('0 days 02:00:00.00000'))].iloc[-1]['datetimestamp']
      val = InsulinData[(InsulinData['datetimestamp']>=nmStart) & (InsulinData['datetimestamp']<=nmEnd)]

      while len(val)>0 and not any(val[carb]>0.0) and nmEnd <= lastInsulineDate:
        no_meal_dates.append(nmStart)
        nmStart = InsulinData[InsulinData['datetimestamp']>nmEnd].iloc[-1]['datetimestamp']

        if (len(InsulinData[InsulinData['datetimestamp']>(nmStart+pd.to_timedelta('0 days 02:00:00.00000'))]))>0:
          nmEnd = InsulinData[InsulinData['datetimestamp']>=(nmStart+pd.to_timedelta('0 days 02:00:00.00000'))].iloc[-1]['datetimestamp']
          val = InsulinData[(InsulinData['datetimestamp']>=nmStart) & (InsulinData['datetimestamp']<=nmEnd)]
        else:
          break

  CGMData = pd.read_csv(CGMFile)
  CGMData['datetimestamp'] = pd.to_datetime(CGMData['Date'] +" "+ CGMData['Time'])
  CGMData = CGMData[['Date', 'Time', 'datetimestamp',sensor]]
  
  noMealData = []
  for val in no_meal_dates:
    end = val+pd.to_timedelta('0 days 02:00:00.00000')
    tempData=CGMData[(CGMData['datetimestamp'] >= val) & (CGMData['datetimestamp']<= end)]
    if len(tempData[tempData['Sensor Glucose (mg/dL)'].isna()]) <= 5 and len(tempData)==24:
      noMealData.append(tempData[::-1])

  return no_meal_dates, noMealData

noMealDates1, noMealData1 = nonMealDataExtraction(mealDates1, inF1, cgmF1)
noMealDates2, noMealData2 = nonMealDataExtraction(mealDates2, inF2, cgmF2)
noMealData = noMealData1+noMealData2


"""### Feature Extractions"""

def featureExtraction(dataList):
  mealFData = []
  for data in dataList:
    data['Sensor Glucose (mg/dL)'] = data['Sensor Glucose (mg/dL)'].interpolate(method = 'linear')
    data['Sensor Glucose (mg/dL)'] = data['Sensor Glucose (mg/dL)'].interpolate(method = 'linear', limit_direction = 'backward', axis = 0)
    featureMap = {}
    # idx = data.iloc[6:]['Sensor Glucose (mg/dL)'].idxmax()
    idx = data['Sensor Glucose (mg/dL)'].idxmax()
    featureMap['tau'] = data.loc[idx, 'datetimestamp'] - data["datetimestamp"].iloc[6]
    featureMap['tau'] = featureMap['tau'].total_seconds()/60

    CGM_meal = data["Sensor Glucose (mg/dL)"].iloc[6]
    dg = data.loc[idx, 'Sensor Glucose (mg/dL)'] - CGM_meal
    # featureMap['dG'] = dg
    dgn = dg/CGM_meal
    featureMap['dGn'] = dgn

    featureMap['diff']=data['Sensor Glucose (mg/dL)'].diff().max()#data.loc[idx, 'Sensor Glucose (mg/dL)'] - data.loc[idx-1, 'Sensor Glucose (mg/dL)']
    #ddata = data['Sensor Glucose (mg/dL)'].diff().diff().max()
    featureMap['ddiff']=data['Sensor Glucose (mg/dL)'].diff().diff().max()

    FFT = abs(np.fft.rfft(data['Sensor Glucose (mg/dL)'])).tolist()
    FFT_sorted = abs(np.fft.rfft(data['Sensor Glucose (mg/dL)'])).tolist()
    FFT_sorted.sort()

    featureMap['pf2'] = FFT_sorted[-2]
    featureMap['f2'] = FFT.index(FFT_sorted[-2])
    featureMap['pf3'] = FFT_sorted[-3]
    featureMap['f3'] = FFT.index(FFT_sorted[-3])

    mealFData.append(featureMap)

  return pd.DataFrame.from_dict(mealFData)


mealDF = featureExtraction(mealData)
mealDF['label'] = 1

noMealDF = featureExtraction(noMealData)
noMealDF['label'] = 0


"""### creating training and test """

def dataCreation():
  mInd, nmInd = int(len(mealDF)*0.8), int(len(noMealDF)*0.8)
  trainMeal, trainNoMeal = mealDF[:mInd], noMealDF[:nmInd]
  trainDF = pd.concat([trainMeal, trainNoMeal])


  valMeal, valNoMeal = mealDF[mInd:], noMealDF[nmInd:]
  valDF = pd.concat([valMeal, valNoMeal])


  train = trainDF[['tau', 'dGn', 'diff', 'ddiff', 'pf2', 'f2', 'pf3', 'f3']]
  tlabel = trainDF[['label']]
  val = valDF[['tau', 'dGn', 'diff', 'ddiff', 'pf2', 'f2', 'pf3', 'f3']]
  vlabels = valDF[['label']]

  return train, tlabel, val, vlabels


def dataCreationTest():

  trainDF = pd.concat([mealDF, noMealDF])
  train = trainDF[['tau', 'dGn', 'diff', 'ddiff', 'pf2', 'f2', 'pf3', 'f3']]
  tlabel = trainDF[['label']]

  return train, tlabel

# trainDF, tlabels, valDF, vlabels = dataCreation()
trainDF, tlabels = dataCreationTest()

"""### Training"""
# def training(X_train,y_train,X_test,y_test):
#   model = RandomForestClassifier(random_state=0)
#   model.fit(X_train,y_train)
#   y_pred = model.predict(X_test)

#   accuracies = cross_val_score(estimator= model, X = X_train, y = y_train, cv=k)
            
#   accuracy = accuracy_score(y_test,y_pred)
#   precision = precision_score(y_test,y_pred)
#   recall = recall_score(y_test,y_pred)
#   f1 = f1_score(y_test,y_pred)


#   filename = "model.sav"
#   filePath = os.path.join(model_dir, filename)
#   pickle.dump(model, open(filePath, 'wb'))

#   print("Model: ","RandomForest" )
#   print("K-Fold Accuracies: ",accuracies )
#   print("Accuracy: ",accuracy )
#   print("Precision: ",precision )
#   print("Recall: ", recall )
#   print("F1 Score': ",f1 )

def training(X_train,y_train):
  model = RandomForestClassifier(random_state=0)
  model.fit(X_train,y_train)
  scores = cross_validate(estimator= model, X = X_train, y = y_train, cv=k, scoring = ['f1','precision','accuracy', 'recall'])

  filename = "model.sav"
  filePath = os.path.join(model_dir, filename)
  pickle.dump(model, open(filePath, 'wb'))

  print("Model: ","RandomForest" )
  print("K-Fold Accuracies: ",scores['test_accuracy'] )
  print("K-Fold Precision: ",scores['test_precision'] )
  print("K-Fold F1: ",scores['test_f1'] )
  print("K-Fold Accuracies: ",scores['test_recall'] )

  print("K-Fold Accuracies: ",scores['test_accuracy'].mean() )
  print("K-Fold Precision: ",scores['test_precision'].mean()  )
  print("K-Fold F1: ",scores['test_f1'].mean()  )
  print("K-Fold Accuracies: ",scores['test_recall'].mean() )


# training(trainDF, tlabels, valDF, vlabels)
training(trainDF, tlabels)
