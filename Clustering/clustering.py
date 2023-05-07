

import pandas as pd
import numpy as np
import math
from scipy.stats import entropy, iqr
from scipy.signal import periodogram
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

carb = "BWZ Carb Input (grams)"
sensor = "Sensor Glucose (mg/dL)"

InsulinData = pd.read_csv('./InsulinData.csv')
InsulinData = InsulinData[['Date', 'Time', carb ]]
InsulinData['datetimestamp'] = pd.to_datetime(InsulinData['Date'] +" "+ InsulinData['Time'])
InsulinData.drop(['Date',"Time"], axis=1, inplace=True)

CGMData = pd.read_csv('./CGMData.csv')
CGMData['datetimestamp'] = pd.to_datetime(CGMData['Date'] +" "+ CGMData['Time'])
CGMData = CGMData[['datetimestamp',sensor]]


def mealDataExtraction():
  meal_filtered = InsulinData[InsulinData[carb].notnull()]
  meal_filtered=meal_filtered[meal_filtered[carb]>0]
  meal_filtered['diff'] = meal_filtered['datetimestamp'][::-1].diff()
  meal_filtered=meal_filtered[::-1]
  meal_filtered['hd']=meal_filtered['diff'] >=  pd.to_timedelta('0 days 02:00:00.00000')
  meal_filtered['flag'] = meal_filtered['hd'].shift(-1)
  meal_filtered['flag'] = meal_filtered['flag'].fillna(True)
  meal_filtered = meal_filtered[meal_filtered['flag']]

  meal_data = meal_filtered[[carb,'datetimestamp']].reset_index(drop=True)
  groundTruth = []
  mealData = []
  groundTruthDF = []

  for row in meal_data.itertuples():

    val = row.datetimestamp
    start = val-pd.to_timedelta('0 days 00:30:00.00000')
    end = val+pd.to_timedelta('0 days 02:00:00.00000')
    tempData=CGMData[(CGMData['datetimestamp']>=start) & (CGMData['datetimestamp']<=end)]
    
    if len(tempData[tempData[sensor].isna()]) <= 6 and len(tempData)>=30:
      # tempData[sensor] = tempData[sensor].interpolate(method="linear", limit_direction="backward", axis=0)
      tempData[sensor] = tempData[sensor].interpolate(method="linear", limit_direction="both", axis=0)
      mealData.append(tempData[::-1])
      groundTruth.append(row._1)
      groundTruthDF.append(row)
  
  df = pd.DataFrame(groundTruthDF)
  df  = df.rename(columns={"_1":carb})
  df = df[[carb]]
  bins = np.arange(df[carb].min(), df[carb].max()+20, 20)
  df['labels'] = pd.cut(df[carb], bins=bins, right = False, labels=[c for c in range(1,len(bins))])
  target = df['labels'].values

  return mealData, df, target

def featureExtraction(dataList):
  mealFData = []
  for data in dataList:

    featureMap = {}
    velocity = data[sensor].diff()
    acceleration = velocity.diff()

    featureMap['vel_mean'] = velocity.mean()
    featureMap['vel_min'] = velocity.min()
    featureMap['vel_max'] = velocity.max()

    featureMap['acc_mean'] = acceleration.mean()
    featureMap['acc_min'] = acceleration.min()
    featureMap['acc_max'] = acceleration.max()

    featureMap['iqr'] = iqr(data[sensor])
    featureMap['entropy'] = entropy(data[sensor])

    _, psd = periodogram(data[sensor])
    featureMap['psd1'] =  psd[0:5].mean()
    featureMap['psd2'] =  psd[5:10].mean()
    featureMap['psd3'] =  psd[10:16].mean()

    sortedfft = sorted(abs(np.fft.rfft(data[sensor])).tolist())

    featureMap['fft1'] = sortedfft[-1]
    featureMap['fft2'] = sortedfft[-2]
    featureMap['fft3'] = sortedfft[-3]
    featureMap['fft4'] = sortedfft[-4]
    featureMap['fft5'] = sortedfft[-5]
    featureMap['fft6'] = sortedfft[-6]
    
    mealFData.append(featureMap)

  return pd.DataFrame.from_dict(mealFData)

def findBinMatrix(preds, combinedFeatures):
  uniquePreds = set(preds)
  uniquePreds.discard(-1)
  binMatrix = [None]*len(uniquePreds)
  for label in uniquePreds:
    binMatrix[label] = combinedFeatures[preds==label]['groundTruth'].value_counts().to_dict()

  binM = pd.DataFrame.from_dict(binMatrix)
  binM = binM.sort_index(axis=1)

  return binM


def findEntropy(predictions):
  entropy = 0.0
  labels = set(predictions)
  for label in labels:
    p = np.sum(predictions == label) / len(predictions)
    if p > 0:
        entropy -= p * np.log2(p)
  
  return entropy

def findPurity(predictions, target):
  labels = set(predictions)
  purity = 0.0
  for label in labels:
      gt_labels = target[predictions == label]
      gt_labels_int = gt_labels.astype(np.int64)
      gt_counts = np.bincount(gt_labels_int.ravel())
      if len(gt_counts) == 0:
        purity_for_label = 0.0
      else:
        purity_for_label = np.max(gt_counts) / np.sum(gt_counts)
      purity += purity_for_label
  purity /= len(labels)
  return purity


def calculate_sse_dbscan(feature_matrix, predicted_labels):
      n_clusters = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
      if n_clusters == 0:
        return 0.0

    # Calculate the centroid for each cluster
      centroids = np.zeros((n_clusters, feature_matrix.shape[1]))
      for i in range(n_clusters):
          cluster_points = feature_matrix[predicted_labels == i]
          if len(cluster_points) == 0:
              centroids[i] = np.zeros(feature_matrix.shape[1])
          else:
              centroids[i] = np.mean(cluster_points, axis=0)

      # Calculate the SSE
      distances = cdist(feature_matrix, centroids, 'euclidean')
      sse = np.sum(np.min(distances, axis=1) ** 2)

      return sse


def kMeans(features, target, combinedFeatures, no_bins):
  # X_train = mealFeatures
  # StandardScaler().fit_transform(X_train)
  # kmeans = KMeans(n_clusters=no_bins, random_state=0, n_init="auto").fit(X_train)
  # print(kmeans.inertia_)
  # predictions = kmeans.labels_

  X_train = features
  StandardScaler().fit_transform(X_train)
  kmeans = KMeans(n_clusters=no_bins, random_state=0, n_init="auto").fit(X_train)
  # print(kmeans.inertia_)
  kmean_pred = kmeans.labels_
  # kmean_pred

  # data = findBinMatrix(kmean_pred, combinedFeatures)
  return kmeans.inertia_, findEntropy(kmean_pred), findPurity(kmean_pred, target), findBinMatrix(kmean_pred, combinedFeatures)


def dbscan(features, target, combinedFeatures, no_bins):
  # model= DBSCAN(eps=1.2, min_samples=5)
  model = DBSCAN(eps=2, min_samples=5)
  model.fit(features)
  db_labels = model.labels_
  db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

  pred_labels = db_labels

  while db_clusters < no_bins:
    # print(db_clusters, no_bins, pred_labels)
    sse_dict = {}
    for label in set(pred_labels):
      if label == -1 or len(features[pred_labels == label]) < 2:
        continue
      sse_dict[label] = calculate_sse_dbscan(features[pred_labels == label], pred_labels[pred_labels == label])
    sorted_labels = [label for label, _ in sorted(sse_dict.items(), key=lambda item: item[1], reverse=True)]
    # print(sorted_labels, sse_dict, sorted(sorted_labels))

    if len(sorted_labels) > 1:
      kmeans = KMeans(n_clusters=2, n_init=10).fit(features[pred_labels == sorted_labels[0]])
      kmean_labels = kmeans.predict(features[pred_labels == sorted_labels[0]])

      newArray = pred_labels[pred_labels == sorted_labels[0]].copy()

      newArray[kmean_labels == 0] = sorted_labels[0]
      newArray[kmean_labels == 1] = sorted(sorted_labels)[-1] + 1

      pred_labels[pred_labels == sorted_labels[0]] = newArray

      db_clusters += 1
    else:
      break

  return calculate_sse_dbscan(features, pred_labels), findEntropy(pred_labels), findPurity(pred_labels, target), findBinMatrix(pred_labels, combinedFeatures)




if __name__ == '__main__':

  mealData, groundTruth, target = mealDataExtraction()
  print(" ----------------------------------- ")

  print("Bin Matrix: ")
  print(target.tolist())
  print(" ----------------------------------- ")

  minValue, maxValue = groundTruth[carb].min(), groundTruth[carb].max()  # min(groundTruth), max(groundTruth)
  no_bins = math.ceil((maxValue - minValue) / 20)
  print("\n Min Value : ", minValue, "\n Max Value : ", maxValue, "\n Number of Bins: ", no_bins)
  print(" ----------------------------------- ")

  mealFeatures = featureExtraction(mealData)
  features = StandardScaler().fit_transform(mealFeatures)
  # features = MinMaxScaler().fit_transform(mealFeatures)
  features = pd.DataFrame(PCA(n_components=6).fit_transform(features))

  combinedFeatures = mealFeatures
  combinedFeatures['groundTruth'] = groundTruth['labels']

  kSse, kEntropy, kPurity, kBinMat = kMeans(features, target, combinedFeatures, no_bins)

  print("KMean Bin Cluster Matrix: (clusters \ bins) ")
  print(kBinMat)
  print(" ----------------------------------- ")
  dSse, dEntropy, dPurity, dBinMat = dbscan(features, target, combinedFeatures, no_bins)

  print("\n DBSCAN Bin Cluster Matrix:  (clusters \ bins) ")
  print(dBinMat)

  print(" ----------------------------------- ")

  data = {"Kmean_SSE": kSse, "dbscan_SSE": dSse,
          "Kmean_Entropy": kEntropy, "dbscan_Entropy": dEntropy,
          "Kmean_Purity": kPurity, "dbscan_Purity": dPurity}

  resultDF = pd.DataFrame.from_dict(data, orient='index')
  print(resultDF)

  resultDF = resultDF.T
  resultDF.to_csv('./Result.csv', header=None, index=True)

  print(" ----------------------------------- ")









  
