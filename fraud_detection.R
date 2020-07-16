#read in credit card data
credit_card <- read.csv("creditcard.csv")

#check structure
str(credit_card)

#look at first/last observations
head(credit_card)
tail(credit_card)

#remove time column
credit_card <- credit_card[,-1]

#check for missing
colSums(is.na(credit_card))

#check out dependent variable
table(credit_card$Class)

#check for zero and near zero features
library(caret)
feature_variance <- nearZeroVar(credit_card[,-30], saveMetrics = TRUE)
feature_variance

#change dependent to factor
credit_card$Class <- as.factor(credit_card$Class)

#split into train and test
set.seed(123)
index <- createDataPartition(credit_card$Class, p = 0.8, list = FALSE)
train <- credit_card[index,]
test <- credit_card[-index,]

#create classification tree
library(rpart)
tree_fit <- rpart(train$Class ~., data = train)
tree_fit$cptable

plotcp(tree_fit)

#install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(tree_fit, type = 3, extra = 2, branch = .75, under = TRUE)

rpart.rules(tree_fit)

testing <- predict(tree_fit, newdata = test)
testing <- testing[,2]

#validate model with AUC
#install.packages("MLmetrics")
ynum <- as.numeric(ifelse(test$Class == "1", 1,0))
library(MLmetrics)
AUC(testing,ynum)

#random forest
x <- as.matrix(train[,-30])
y <- as.factor(train$Class)


#install.packages("randomForest")
library(randomForest)
set.seed(123)
forest_fit <- randomForest(x = x, y = y, ntree = 200)
forest_fit

min <- which.min(forest_fit$err.rate[,1])
forest_fit$err.rate[min]

varImpPlot(forest_fit)

ff <- data.frame(unlist(forest_fit$importance))
ff$var <- row.names(ff)

summary(ff)

rf_prob <- predict(forest_fit, type = "prob")
y_prob <- rf_prob[,2]
density_plot(y, y_prob)
ynum <- as.numeric(ifelse(y == "1", 1,0))
AUC(y_prob, ynum)
LogLoss(y_prob, ynum)

rf_test <- predict(forest_fit, type = "prob", newdata = test)
rf_test <- rf_test[,2]
ytest <- as.numeric(ifelse(test$Class == "1", 1,0))
AUC(rf_test, ytest)

#comparing models
r_part_roc <- ROCR::prediction(testing, test$Class)
r_part_roc2 <- ROCR::performance(r_part_roc, "tpr", "fpr")
ROCR::plot(r_part_roc2, col = 1)

rf_roc <- ROCR::prediction(rf_test, test$Class)
rf_roc <- ROCR::performance(rf_roc, "tpr", "fpr")
ROCR::plot(rf_roc, col = 2, add = TRUE)
legend(0.6,0.6, c("rpart", "rf"), 1:2)
