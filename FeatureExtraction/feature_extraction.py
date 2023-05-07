
import pandas as pd
import numpy as np

dropDates = .8*288
manualCode = "AUTO MODE ACTIVE PLGM OFF"
percentTotal = 288
hyperGla = 180
hyperGlaC = 250
rangeStart = 70
rangeEnd = 180
rangeSecStart = 70
rangeSecEnd = 150
hypoGla1 = 70
hypoGla2 = 54


hyperStr = "hyper"
hyperCStr = "hyperc"
rangeStr = "r"
rangeSStr = "rs"
hypol1Str = "hypol1"
hypol2Str = "hypol2"

CGMData = pd.read_csv('./CGMData.csv')
CGMData.head()

CGMData = CGMData[['Date', 'Time','Sensor Glucose (mg/dL)']]
CGMData['datetimestamp'] = pd.to_datetime(CGMData['Date'] +" "+ CGMData['Time'])
#CGMData.drop(['Date',"Time"], axis=1, inplace=True)
CGMData.drop(CGMData[CGMData['Sensor Glucose (mg/dL)'].isna()].index, inplace = True)

# CGMData.head()
# CGMData.shape

grouped = CGMData.groupby(['Date']).size().reset_index(name='counts')
filtered = grouped[grouped['counts'] > dropDates]
result = CGMData[CGMData.apply(lambda x: (x['Date']) in filtered[['Date']].values, axis=1)]
CGMData=result
# CGMData.shape
CGMData.set_index('datetimestamp')


InsulinData = pd.read_csv('./InsulinData.csv')
InsulinData = InsulinData[['Date', 'Time','Alarm']]
InsulinData['datetimestamp'] = pd.to_datetime(InsulinData['Date'] +" "+ InsulinData['Time'])
InsulinData.drop(['Date',"Time"], axis=1, inplace=True)
#InsulinData.head()

#
# def dataAnalysis(prefix, data):
#   days = data.groupby('Date').count().shape[0]
#   map = {}
#   map[prefix+hyper]=sum(list(data[data['Sensor Glucose (mg/dL)']>hyperGla].groupby(['Date'])['Sensor Glucose (mg/dL)'].count()*100/percentTotal))/days
#   map[prefix+hyperC]=sum(list(data[data['Sensor Glucose (mg/dL)']>hyperGlaC].groupby(['Date'])['Sensor Glucose (mg/dL)'].count()*100/percentTotal))/days
#   map[prefix+range]=sum(list(data[(data['Sensor Glucose (mg/dL)']>=rangeStart) & (data['Sensor Glucose (mg/dL)']<= rangeEnd)].groupby(['Date'])['Sensor Glucose (mg/dL)'].count()*100/percentTotal))/days
#   map[prefix+rangeS]=sum(list(data[(data['Sensor Glucose (mg/dL)']>=rangeSecStart) & (data['Sensor Glucose (mg/dL)']<= rangeSecEnd)].groupby(['Date'])['Sensor Glucose (mg/dL)'].count()*100/percentTotal))/days
#   map[prefix+hypol1]=sum(list(data[data['Sensor Glucose (mg/dL)']<hypoGla1].groupby(['Date'])['Sensor Glucose (mg/dL)'].count()*100/percentTotal))/days
#   map[prefix+hypol2]=sum(list(data[data['Sensor Glucose (mg/dL)']<hypoGla2].groupby(['Date'])['Sensor Glucose (mg/dL)'].count()*100/percentTotal))/days
#   return map

def dataAnalysis(dataA):
  data = dataA
  hyper = data[data['Sensor Glucose (mg/dL)'] > hyperGla]
  hyperC = data[data['Sensor Glucose (mg/dL)'] > hyperGlaC]
  range = data[(data['Sensor Glucose (mg/dL)'] >= rangeStart) & (data['Sensor Glucose (mg/dL)'] <= rangeEnd)]
  rangeS = data[(data['Sensor Glucose (mg/dL)'] >= rangeSecStart) & (data['Sensor Glucose (mg/dL)'] <= rangeSecEnd)]
  hypol1 = data[data['Sensor Glucose (mg/dL)'] < hypoGla1]
  hypol2 = data[data['Sensor Glucose (mg/dL)'] < hypoGla2]

  hd = hyper['Date'].unique().shape[0]
  hCd = hyperC['Date'].unique().shape[0]
  rd = range['Date'].unique().shape[0]
  rSd = rangeS['Date'].unique().shape[0]
  hlpd1 = hypol1['Date'].unique().shape[0]
  hlpd2 = hypol2['Date'].unique().shape[0]

  hyperN = hyper.between_time('00:00:00', '05:59:59')
  hyperCN = hyperC.between_time('00:00:00', '05:59:59')
  rangeN = range.between_time('00:00:00', '05:59:59')
  rangeSN = rangeS.between_time('00:00:00', '05:59:59')
  hypol1N = hypol1.between_time('00:00:00', '05:59:59')
  hypol2N = hypol2.between_time('00:00:00', '05:59:59')

  hdN = hyperN['Date'].unique().shape[0]
  hCdN = hyperCN['Date'].unique().shape[0]
  rdN = rangeN['Date'].unique().shape[0]
  rSdN = rangeSN['Date'].unique().shape[0]
  hlpd1N = hypol1N['Date'].unique().shape[0]
  hlpd2N = hypol2N['Date'].unique().shape[0]

  hyperD = hyper.between_time('06:00:00', '23:59:59')
  hyperCD = hyperC.between_time('06:00:00', '23:59:59')
  rangeD = range.between_time('06:00:00', '23:59:59')
  rangeSD = rangeS.between_time('06:00:00', '23:59:59')
  hypol1D = hypol1.between_time('06:00:00', '23:59:59')
  hypol2D = hypol2.between_time('06:00:00', '23:59:59')

  hdD = hyperD['Date'].unique().shape[0]
  hCdD = hyperCD['Date'].unique().shape[0]
  rdD = rangeD['Date'].unique().shape[0]
  rSdD = rangeSD['Date'].unique().shape[0]
  hlpd1D = hypol1D['Date'].unique().shape[0]
  hlpd2D = hypol2D['Date'].unique().shape[0]

  # print(hlpd2N)
  map = {}
  prefix = "A "
  map[prefix + hyperStr] = np.sum(hyperN.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hdN
  map[prefix + hyperCStr] = np.sum(
    hyperCN.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hCdN
  map[prefix + rangeStr] = np.sum(rangeN.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / rdN
  map[prefix + rangeSStr] = np.sum(
    rangeSN.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / rSdN
  map[prefix + hypol1Str] = np.sum(
    hypol1N.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hlpd1N
  map[prefix + hypol2Str] = np.sum(
    hypol2N.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hlpd2N

  prefix = "B "
  map[prefix + hyperStr] = np.sum(hyperD.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hdD
  map[prefix + hyperCStr] = np.sum(
    hyperCD.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hCdD
  map[prefix + rangeStr] = np.sum(rangeD.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / rdD
  map[prefix + rangeSStr] = np.sum(
    rangeSD.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / rSdD
  map[prefix + hypol1Str] = np.sum(
    hypol1D.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hlpd1D
  map[prefix + hypol2Str] = np.sum(
    hypol2D.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hlpd2D

  # print(hypol2D.groupby(['Date'])['Sensor Glucose (mg/dL)'].count()*100/percentTotal, hypol2D['Date'].unique().shape[0])

  prefix = "C "
  map[prefix + hyperStr] = np.sum(hyper.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hd
  map[prefix + hyperCStr] = np.sum(
    hyperC.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hCd
  map[prefix + rangeStr] = np.sum(range.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / rd
  map[prefix + rangeSStr] = np.sum(
    rangeS.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / rSd
  map[prefix + hypol1Str] = np.sum(
    hypol1.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hlpd1
  map[prefix + hypol2Str] = np.sum(
    hypol2.groupby(['Date'])['Sensor Glucose (mg/dL)'].count() * 100 / percentTotal) / hlpd2

  return map

manualTimeStampStop = InsulinData[InsulinData["Alarm"]==manualCode].iloc[-1]
mData= CGMData[CGMData['datetimestamp']<manualTimeStampStop['datetimestamp']]
mData=mData.set_index(pd.DatetimeIndex(mData['datetimestamp']))

aData= CGMData[CGMData['datetimestamp']>=manualTimeStampStop['datetimestamp']]
aData=aData.set_index(pd.DatetimeIndex(aData['datetimestamp']))

assert aData.shape[0]+mData.shape[0] == CGMData.shape[0]

# data = mData
# mDayTimeData = dataAnalysis("B ", data.between_time('06:00:00', '23:59:59'))
# nightData = dataAnalysis("A ", data.between_time('00:00:00', '05:59:59'))
# allDayData = dataAnalysis("c ", data)
# nightData.update(mDayTimeData)
# nightData.update(allDayData)

mRow=dataAnalysis(mData)
aRow = dataAnalysis(aData)

dataFrame =pd.DataFrame()
dataFrame = dataFrame.append(mRow, ignore_index=True)
dataFrame=dataFrame.append(aRow, ignore_index=True)
dataFrame = dataFrame.fillna(0.0)

dataFrame.to_csv('Result.csv', header=False, index=False,index_label=['Manual Mode','Auto Mode'])