# Practical Machine Learning: Course Project Writeup-Human Activity Recognition
JMST  
December 14, 2015  

##Overview
  Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  
  
  In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.  
  
  The goal of our project is to predict the manner in which exercise was done. This is the "classe" variable in the training set. I may use any of the other variables to predict with. I am expected to create a report describing how to built the model, how cross validation will be used, what I think the expected out of sample error is, and why I made the choices I did. I will also use my prediction model to predict 20 different test cases.
  
##Loading and preprocessing the data

The training data for this project are available here:
(https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:
(https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

The data for this project come from this source: 
(http://groupware.les.inf.puc-rio.br/har). 

Setting Directory and Loading required packages:

```
## Warning: package 'ggplot2' was built under R version 3.2.2
```

```
## Warning: package 'caret' was built under R version 3.2.3
```

```
## Warning: package 'survival' was built under R version 3.2.3
```

```
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Warning: package 'cluster' was built under R version 3.2.3
```

```
## Warning: package 'plyr' was built under R version 3.2.2
```

##Loading and reading data
The training set consists of 19622 observations of 160 variables, one of which is the dependent variable as far as this study is concerned.
Irrelevant variables can be removed as they are unlikely to be related to dependent variable 

```r
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
tData = read.csv("pml-training.csv", na.strings=c("", "NA", "NULL"))
testDt = read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL"))
dim(tData)
```

```
## [1] 19622   160
```

```r
dim(testDt)
```

```
## [1]  20 160
```

```r
trainingDt <- tData[ , colSums(is.na(tData)) == 0]
dim(trainingDt)
```

```
## [1] 19622    60
```

```r
DtNAs <- apply(tData, 2, function(x) { sum(is.na(x)) })
LegitDt <- subset(tData[, which(DtNAs == 0)], 
                     select=-c(X, user_name, new_window, num_window, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
dim(LegitDt)
```

```
## [1] 19622    53
```

```r
testDt <- testDt[, colSums(is.na(testDt)) == 0]
testData <- testDt[, -c(1:7)]
dim(testData)
```

```
## [1] 20 53
```

##Splitting Data to create Model

We now split the updated training dataset into a training dataset (80% of the observations) and a validation dataset (20% of the observations). This validation dataset will allow us to perform cross validation when developing our model.

```r
teaching <- createDataPartition(LegitDt$classe, p=0.8, list=F)
tData <- LegitDt[teaching,]
testDt <- LegitDt[-teaching,]
```

##Cross-validation

We will use random forest as our model as implemented in the randomForest package by Breiman's random forest algorithm (based on Breiman and Cutler's original Fortran code) for classification and regression.Using random forest, the out of sample error should be small.

```r
Dom <- trainControl(method = "cv", number = 5)
 RpartDt <- train(classe ~ ., data = tData, method = "rpart", 
                    trControl = Dom)
```

```
## Loading required package: rpart
```

```
## Warning: package 'rpart' was built under R version 3.2.3
```

```r
print(RpartDt, digits = 4)
```

```
## CART 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 12559, 12557, 12561, 12560, 12559 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa    Accuracy SD  Kappa SD
##   0.03543  0.5024    0.34966  0.01088      0.01432 
##   0.06100  0.4424    0.25204  0.07129      0.11893 
##   0.11571  0.3163    0.04872  0.04385      0.06676 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03543.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.3
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
mock0 <- randomForest(classe ~ ., data = tData)
mock0
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = tData) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.44%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4462    1    0    0    1 0.0004480287
## B   13 3020    5    0    0 0.0059249506
## C    0   12 2723    3    0 0.0054784514
## D    0    0   22 2549    2 0.0093276331
## E    0    0    2    8 2876 0.0034650035
```

```r
mock0$confusion
```

```
##      A    B    C    D    E  class.error
## A 4462    1    0    0    1 0.0004480287
## B   13 3020    5    0    0 0.0059249506
## C    0   12 2723    3    0 0.0054784514
## D    0    0   22 2549    2 0.0093276331
## E    0    0    2    8 2876 0.0034650035
```

```r
crossprognosis <- predict(mock0, testDt)
#Checking the predictions against the data held from the test data.
sum(crossprognosis == testDt$classe) / length(crossprognosis)
```

```
## [1] 0.997196
```

```r
confusionMatrix(testDt$classe, crossprognosis)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  759    0    0    0
##          C    0    3  679    2    0
##          D    0    0    4  639    0
##          E    0    0    0    2  719
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9972         
##                  95% CI : (0.995, 0.9986)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9965         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9961   0.9941   0.9938   1.0000
## Specificity            1.0000   1.0000   0.9985   0.9988   0.9994
## Pos Pred Value         1.0000   1.0000   0.9927   0.9938   0.9972
## Neg Pred Value         1.0000   0.9991   0.9988   0.9988   1.0000
## Prevalence             0.2845   0.1942   0.1741   0.1639   0.1833
## Detection Rate         0.2845   0.1935   0.1731   0.1629   0.1833
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   0.9980   0.9963   0.9963   0.9997
```
our model shows a at least 99.72% accuracy against our test dataset and this is confirmed by the above confusion matrix. Out of sample accuracy estimated at 0.9972 , or 99.72% So that means the out-of-sample error We estimate is about 0.0028, or 0.28%. The out-of-sample error should not be over 0.5% for most accurate results.

##Models and Predictions


```r
#Let us take a look at Random forest Model VS rpart Model
#Ramdom Forest
set.seed(1111)
library(randomForest)
mock1 <- randomForest(classe ~. , data=tData, method="class")
prognosis1 <- predict(mock1, testDt, type = "class")
confusionMatrix(prognosis1, testDt$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    1    0    0    0
##          B    0  758    3    0    0
##          C    0    0  679    5    0
##          D    0    0    2  638    2
##          E    0    0    0    0  719
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9967          
##                  95% CI : (0.9943, 0.9982)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9958          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9987   0.9927   0.9922   0.9972
## Specificity            0.9996   0.9991   0.9985   0.9988   1.0000
## Pos Pred Value         0.9991   0.9961   0.9927   0.9938   1.0000
## Neg Pred Value         1.0000   0.9997   0.9985   0.9985   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1932   0.1731   0.1626   0.1833
## Detection Prevalence   0.2847   0.1940   0.1744   0.1637   0.1833
## Balanced Accuracy      0.9998   0.9989   0.9956   0.9955   0.9986
```

```r
#rpart Model
library(rpart)
mock2 <- rpart(classe ~ ., data=tData, method="class")
prognosis2 <- predict(mock2, testDt, type = "class")
confusionMatrix(prognosis2, testDt$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 990 147  10  60  23
##          B  30 433  44  17  49
##          C  35  79 562  91 106
##          D  40  60  46 423  46
##          E  21  40  22  52 497
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7405          
##                  95% CI : (0.7265, 0.7542)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6709          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8871   0.5705   0.8216   0.6579   0.6893
## Specificity            0.9145   0.9558   0.9040   0.9415   0.9578
## Pos Pred Value         0.8049   0.7557   0.6438   0.6878   0.7864
## Neg Pred Value         0.9532   0.9027   0.9600   0.9335   0.9319
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2524   0.1104   0.1433   0.1078   0.1267
## Detection Prevalence   0.3135   0.1461   0.2225   0.1568   0.1611
## Balanced Accuracy      0.9008   0.7631   0.8628   0.7997   0.8236
```

##Conclusion

As it is seen, the confusion matrix from the Random Forest model is very accurate. I tested with different models but only the Random Forest model turned out to be the most accurate for this dataset. Because my test data was around 99% accurate I expected nearly all of the submitted test cases to be correct. All my 20 tested cases turned were correct at submission.

##Submission to Coursera


```r
answers=predict(mock1,testData)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
pml_write_files(answers)
```


