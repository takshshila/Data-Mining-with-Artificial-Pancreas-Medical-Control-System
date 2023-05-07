
import pandas as pd
import numpy as np
from numpy.fft import fft2
import os
import pickle

model_dir = "./" #"./models"
filename = 'model.sav'
filePath = os.path.join(model_dir, filename)
saved_model = pickle.load(open(filePath, 'rb'))


def prediction(dataRow):
  data = pd.DataFrame({'val': dataRow})
  featureMap = {}
  mealData =[]
  idx = data['val'].idxmax()
  featureMap['tau'] = idx * 5.0
  
  dg = data['val'].loc[idx] - data['val'].iloc[0]
  # featureMap['dG'] = dg
  dgn = dg/data['val'].iloc[0]
  featureMap['dGn'] = dgn

  featureMap['diff']=data['val'].diff().max()
  featureMap['ddiff']=data['val'].diff().diff().max()

  FFT = abs(np.fft.rfft(data['val'])).tolist()
  FFT_sorted = abs(np.fft.rfft(data['val'])).tolist()
  FFT_sorted.sort()

  featureMap['pf2'] = FFT_sorted[-2]
  featureMap['f2'] = FFT.index(FFT_sorted[-2])
  featureMap['pf3'] = FFT_sorted[-3]
  featureMap['f3'] = FFT.index(FFT_sorted[-3])

  mealData.append(featureMap)

  input = pd.DataFrame.from_dict(mealData)
  preds = saved_model.predict(input)

  return preds[0]


testData = pd.read_csv("./test.csv", header = None)
result = []
for id, data in testData.iterrows():
  cPred = prediction(data)
  # print(cPred)
  result.append(cPred)

df = pd.DataFrame(result)
df.to_csv('result.csv', index=False, header = False)




# testData = pd.read_csv("./test.csv", header = None)
# testDataList = []
# for id, data in testData.iterrows():
#   testDataList.append(pd.DataFrame({'val': data}))


# def testResult():
#   result = []
#   for data in testDataList:
#     featureMap = {}
#     mealData =[]
#     idx = data['val'].idxmax()
#     featureMap['tau'] = idx * 5.0
    
#     dg = data['val'].loc[idx] - data['val'].iloc[0]
#     # featureMap['dG'] = dg
#     dgn = dg/data['val'].iloc[0]
#     featureMap['dGn'] = dgn

#     featureMap['diff']=data['val'].diff().max()
#     featureMap['ddiff']=data['val'].diff().diff().max()

#     FFT = abs(np.fft.rfft(data['val'])).tolist()
#     FFT_sorted = abs(np.fft.rfft(data['val'])).tolist()
#     FFT_sorted.sort()

#     featureMap['pf2'] = FFT_sorted[-2]
#     featureMap['f2'] = FFT.index(FFT_sorted[-2])
#     featureMap['pf3'] = FFT_sorted[-3]
#     featureMap['f3'] = FFT.index(FFT_sorted[-3])

#     mealData.append(featureMap)

#     input = pd.DataFrame.from_dict(mealData)
#     preds = saved_model.predict(input)
#     result.append(preds[0])
    
#   return result

# result = testResult()

# df = pd.DataFrame(result)
# df.to_csv('result.csv', index=False)







