library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(knitr)

set.seed(12345)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

# partion data
inTrain <- createDataPartition(training$classe, p=0.75, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]
dim(myTraining); dim(myTesting)

# clean data
nzv <- nearZeroVar(myTraining, saveMetrics=TRUE)
myTraining <- myTraining[,nzv$nzv==FALSE]

nzv<- nearZeroVar(myTesting,saveMetrics=TRUE)
myTesting <- myTesting[,nzv$nzv==FALSE]

# remove 1st column from data
myTraining <- myTraining[c(-1)]

# remove variables with more than 70% N/A
trainingV3 <- myTraining
for(i in 1:length(myTraining)) {
    if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .7) {
        for(j in 1:length(trainingV3)) {
            if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) == 1)  {
                trainingV3 <- trainingV3[ , -j]
            }   
        } 
    }
}

# Set back to the original variable name
myTraining <- trainingV3
rm(trainingV3)

# transform testing and training datasets
clean1 <- colnames(myTraining)
clean2 <- colnames(myTraining[, -58])  # remove the classe column
myTesting <- myTesting[clean1]         # allow only variables in myTesting that are also in myTraining
testing <- testing[clean2]             # allow only variables in testing that are also in myTraining

dim(myTesting)

# coerce data into the same type ==> NOT NECESSARY
for (i in 1:length(testing) ) {
    for(j in 1:length(myTraining)) {
        if( length( grep(names(myTraining[i]), names(testing)[j]) ) == 1)  {
            class(testing[j]) <- class(myTraining[i])
        }      
    }      
}

# To get the same class between testing and myTraining
testing <- rbind(myTraining[2, -58] , testing)
testing <- testing[-1,]

# Prediction using Decision Trees
set.seed(12345)
modFitA1 <- rpart(classe ~ ., data=myTraining, method="class")
predFitA1 <- predict(modFitA1,myTesting,type='class')

modFitA1 <- train(classe ~ ., data=myTraining, method="rpart")
predFitA1 <- predict(modFitA1,myTesting)

cmatrixA1 <- confusionMatrix(predFitA1,myTesting$classe)
cmatrixA1

require(partykit)
plot(as.party(modFitA1))
fancyRpartPlot(modFitA1)
