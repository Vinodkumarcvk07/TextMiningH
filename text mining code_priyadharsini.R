#----------------------Building and Saving Model--------------------------------------#

#initialization
#------------------------
rm(list=ls())

setwd("E:/data science/project/text mining project")
getwd()

library(ggplot2)
library(data.table)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
library(randomForest)
library(dplyr)
library(nnet)



#load file
#---------------------------
xyztext.raw=read.csv("TextClassification_Data.csv", header = TRUE, stringsAsFactors = FALSE)

#data cleanup
#---------------------------
xyztext.raw=xyztext.raw[,-3]
xyztext.raw=xyztext.raw[,-1]
names(xyztext.raw)=c("summary","categories","sub_categories","previous_appointment","id")

#according to doc specification only 5 categories, so 6th category JUNK have to be removed
xyztext.raw_nojunk=xyztext.raw[!(xyztext.raw$categories=="JUNK"),]


xyztext.raw_nojunk$categories=toupper(xyztext.raw_nojunk$categories)
#xyztext.raw_nojunk[(xyztext.raw_nojunk$categories=="mISCELLANEOUS"),]
xyztext.raw_nojunk$sub_categories=toupper(xyztext.raw_nojunk$sub_categories)
#xyztext.raw_nojunk[(xyztext.raw_nojunk$categories=="ask_A_DOCTOR"),]

length(which(!complete.cases(xyztext.raw_nojunk)))
xyztext.raw_nojunk$categories=as.factor(xyztext.raw_nojunk$categories)
xyztext.raw_nojunk$sub_categories=as.factor(xyztext.raw_nojunk$sub_categories)


#understanding data
#--------------------------
prop.table(table(xyztext.raw_nojunk$categories))
prop.table(table(xyztext.raw_nojunk$sub_categories))

#distribution of text length in description
xyztext.raw_nojunk$Summary_length=nchar(xyztext.raw_nojunk$summary)
print("Length of characters -")
summary(xyztext.raw_nojunk$Summary_length)

#visualization
ggplot(xyztext.raw_nojunk, aes(x=xyztext.raw_nojunk$Summary_length, 
                               fill=categories))+theme_bw()+
  geom_histogram(binwidth = 5)+
  labs(y="Text count",x="Length of text", 
       title="Distribution of lengths with class categories")

#Multivariate Scatter plot
xyztext.raw_nojunk$categories=as.factor(xyztext.raw_nojunk$categories)
xyztext.raw_nojunk$sub_categories=as.factor(xyztext.raw_nojunk$sub_categories)
ggplot(xyztext.raw_nojunk,aes_string(x=xyztext.raw_nojunk$categories,y=xyztext.raw_nojunk$sub_categories))+geom_point(aes_string(colour=xyztext.raw_nojunk$categories),size=4)+theme_bw()+xlab("Categories")+ylab("Sub Categories")+ggtitle("Distribution of categories and sub categories")+theme(text=element_text(size=6))+scale_colour_discrete(name="Categories")+scale_shape_discrete(name="Sub categories")


#splitting data
#-----------------------------
set.seed(32984)
indexes <- createDataPartition(xyztext.raw_nojunk$categories, times = 1,
                               p = 0.2, list = FALSE)
train=xyztext.raw_nojunk[indexes,]
test=xyztext.raw_nojunk[-indexes,]

prop.table(table(train$categories))
prop.table(table(test$categories))


#term document matrix
#----------------------------------
# Create corpus
library(tm)
docs <- Corpus(VectorSource(train$summary))

# Clean corpus
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, stemDocument, language = "english")

#feature engineering
#-------------------------------------
# Create dtm
#dtm <- DocumentTermMatrix(docs)

#tf-idf 
dtm <-DocumentTermMatrix(docs,control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))

# #bigram
# install.packages("RWeka")
# library(RWeka)
# #bigram function
# Bigram_Tokenizer <- function(x){ NGramTokenizer(x, Weka_control(min=2, max=2))}
# dtm <- DocumentTermMatrix(docs, control = list(tokenize = Bigram_Tokenizer))

#feature extraction
#------------------------------
#remove sparse
new_docterm_corpus <- removeSparseTerms(dtm,sparse = 0.9938)

colS <- colSums(as.matrix(new_docterm_corpus))

doc_features <- data.table(name = attributes(colS)$names, count = colS)

library(wordcloud)
wordcloud(names(colS), colS, min.freq = 100, scale = c(6,.1), colors = brewer.pal(6, 'Dark2'))

#create data set for training
#------------------------------------
processed_data <- as.data.table(as.matrix(new_docterm_corpus))

#combing the data
train_data_categ <- cbind(data.table(categories = train$categories, subcategories=train$sub_categories),processed_data)
#att.scores <- random.forest.importance(categories ~ ., train)

#multinominal logistic reg
set.seed(48743)

train_data_categ$categories <- as.factor(train_data_categ$categories)
train_data_categ$out <- relevel(train_data_categ$categories, ref=1)

train_data_categ$subcategories=as.factor(train_data_categ$subcategories)
train_data_categ$subcategop=relevel(train_data_categ$subcategories, ref=1)

index <- createDataPartition(train_data_categ$categories, times = 1,
                               p = 0.7, list = FALSE)
train=train_data_categ[index,]
test=train_data_categ[-index,]

train_data_categ=train

#model building
#------------------------

mnominal_logmodel_categ <- multinom(out~., data=train_data_categ)
mnominal_logmodel_subcateg=multinom(subcategop~., data=train_data_categ[1:1000,50:ncol(train_data_categ)])


library(e1071)
mnB_categ=naiveBayes(train_data_categ, select=-categories, train_data_categ$categories, laplace = 1)
mnB_subcateg=naiveBayes(train_data_categ, select=-subcategories, train_data_categ$subcategories, laplace = 1)

require(mgcv)
saveRDS(mnominal_logmodel_categ, file="category_logistic_model.rda")
saveRDS(mnominal_logmodel_subcateg,file="subcategory_logistic_model.rda")
saveRDS(mnB_categ,file="categ_naive_bayes_model.rda")
saveRDS(mnB_subcateg,file="subcateg_naive_bayes_model.rda")

#--------------------------------------------------------------------------#
        #-------------Load and validate model-----------------#
#---------------------------------------------------------------------------#


#load model
categ_model <- readRDS(file="category_logistic_model.rda")
subcateg_model <- readRDS(file="subcategory_logistic_model.rda")
categ_nB_model <- readRDS(file="categ_naive_bayes_model.rda")
subcateg_nB_model <- readRDS(file="subcateg_naive_bayes_model.rda")

#predict
#Multinomial Logistic Regression 
pred <- predict(categ_model,test,type = "prob")
cm_categ <- table(predict(categ_model),train_data_categ$categories)

pred <- predict(subcateg_model,train_data_categ,type = "prob")
cm_subcateg <- table(predict(subcateg_model),train_data_categ$subcategories[1:1000])

#Naive Bayes model
nBpred=predict(categ_nB_model,train_data_categ,select= -categories)
nB_cm_categ <- table(nBpred,train_data_categ$categories)

nBpred=predict(subcateg_nB_model,train_data_categ,select= -subcategories)
nB_cm_subcateg <- table(nBpred,train_data_categ$subcategories)

#Accuracy
#Multinomial Logistic Regression
print("Multinomial Logistic Regression")
accuracy_categ=sum(diag(cm_categ))/sum(cm_categ)
print("Accuracy of category prediction is-")
print(accuracy_categ)
accuracy_subcateg=sum(diag(cm_subcateg))/sum(cm_subcateg)  #misclassification percentage error
print("Accuracy of sub category prediction is-")
print(accuracy_subcateg)

print("Multiclass naive bayes")
accuracy_categ=sum(diag(nB_cm_categ))/sum(nB_cm_categ)
print("Accuracy of category prediction is-")
print(accuracy_categ)
accuracy_subcateg=sum(diag(nB_cm_subcateg))/sum(nB_cm_subcateg)  #misclassification percentage error

#-------------------------------------------------------------------------#
    #----------Recommendations and develop insights------------------#
#-------------------------------------------------------------------------#

op_categ=as.data.frame.matrix(cm_categ)
op_categ=cbind(op_categ,rownames(op_categ))
colnames(op_categ)[nrow(op_categ)+1]="Categories"

nB_op_categ=as.data.frame.matrix(nB_cm_categ)
nB_op_categ=cbind(nB_op_categ,rownames(nB_op_categ))
colnames(nB_op_categ)[nrow(nB_op_categ)+1]="Categories"

library(plotly)
library(tidyr)
mNL_Categ_plot <- plot_ly(op_categ, x= ~Categories, y = ~APPOINTMENTS, type = 'bar', name = 'Appointments') %>%
  add_trace(y = ~ASK_A_DOCTOR, name = 'Ask a doctor') %>% 
  add_trace(y = ~LAB, name = 'Lab') %>%
  add_trace(y = ~MISCELLANEOUS, name = 'Miscellaneous') %>%
  add_trace(y = ~PRESCRIPTION, name = 'Prescription') %>%
  layout(yaxis = list(title = 'Predicted Value'), barmode = 'group')
mNL_Categ_plot

mNB_Categ_plot <- plot_ly(nB_op_categ, x= ~Categories, y = ~APPOINTMENTS, type = 'bar', name = 'Appointments') %>%
  add_trace(y = ~ASK_A_DOCTOR, name = 'Ask a doctor') %>% 
  add_trace(y = ~LAB, name = 'Lab') %>%
  add_trace(y = ~MISCELLANEOUS, name = 'Miscellaneous') %>%
  add_trace(y = ~PRESCRIPTION, name = 'Prescription') %>%
  layout(yaxis = list(title = 'Predicted Value'), barmode = 'group') %>%
  layout(xaxis = list(title = 'Categories'))
mNB_Categ_plot


out <- capture.output(op_categ)
cat("Multinomial Logistic Regression", out, file="Multinomial_regression_output.txt", sep="n", append=TRUE)
nBout <- capture.output(nB_op_categ)
cat("Naive Bayes output", nBout, file="Naive_Bayes_output.txt", sep="n", append=TRUE)
