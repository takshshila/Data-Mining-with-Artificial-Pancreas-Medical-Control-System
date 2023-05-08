# Data-Mining-with-Artificial-Pancreas-Medical-Control-System
<!-- Performed Feature Extraction, Feature Analysis, Classification, and Clustering -->
<!-- The project worked on data collected from sensor data  -->

In this project, we are considering the Artificial Pancreas medical control system, specifically the Medtronic 670G system. The Medtronic system consists of a continuous glucose monitor (CGM) and the Guardian Sensor (Insuline), which is used to collect blood glucose measurements every 5 minutes.

## Feature Extraction
Synchronised data from Continuous Glucose Sensor( CGMData.csv) and Insulin Pump( InsulinData.csv).

- Percentage time in hyperglycemia (CGM > 180 mg/dL), 
- percentage of time in hyperglycemia critical (CGM > 250 mg/dL), 
- percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL), 
- percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL), 
- percentage time in hypoglycemia level 1 (CGM < 70 mg/dL), and 
- percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).

## Classification
#### Extract meal data and no-meal data by combining InsulinData.csv and CGMData.csv. 
- Meal Data: From the InsulinData.csv, search for non-zero **BWZ Carb Input (grams)** value. This time indicates the start of meal consumption time tm. Meal data comprises of  2hr 30 min stretch of CGM data that starts from tm-30min and extends to tm+2hrs.

- No Meal Data: No meal data comprises 2 hrs of raw data that does not have meal intake. 


#### Feature Engineearing
Calculate features from the both meal and non-meal data. 
- Tau 
- dGN 
- Max value difference (velocity) 
- Max value difference of difference (Acceleration)
- Top 2 Fourier transform values 
- Indexes of top 2 Fourier transform values

#### Model Training
- Merge the both meal data and non-meal data together with their labels, shuffle them. 
- Divide the data into training and validation data.
- Use any Statistical Machine Learning classification model to train datat. 
- Apply same feature engineering techniques to the test data and test the trained model with the test data. 


## Cluster Validation (K-Means and DBSCAN)

#### Extract meal data by combining InsulinData.csv and CGMData.csv. 
Use the same strategy ,as used in classification, for meal data extraction. 

#### Calculate cluster bins as a ground truth.
- Calculate minimum and maximum **BWZ Carb Input (grams)** 
- Make equal sized bins of size 20. Like Bin 0: min_carb, min_carb+20, Bin 1: min_carb+21, min_carb+40, etc. 
- Label data with respect to the bin number they fall into. 

#### Feature Engineearing
Calculate features from the both meal and non-meal data. 
- velocity - 3 columns will be the "first_differential" and consider min, max and mean
- acceleration - 3 columns will be the "second_differential" and consider min, max and avg
-  Entropy - 1 column - use the inbuilt from scipy.stats import entropy
- iqr -1 column - use inbuilt function scipy.stats.iqr 
- fft - 6 columns - apply fft and consider the top 6 max values resulting in 6 columns.
- psd - 3 columns-  you can use inbuilt function scipy.signal.periodogram and take the mean of first 0:5 for psd1 5:10 for psd2 10:16 for psd3

#### Clustering Validation
- Perform clustering K-means on non-labelled data. 
- Perform DBSCAN and find out number of clusters formed. If number of clusters less than the required number of clusters, use bisecting K-Means to get required number of clusters. 
- Calculate SSE, Entropy, and purity for both K-mean and DBSCAN results. 