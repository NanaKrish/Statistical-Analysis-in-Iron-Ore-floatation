#
install.packages("dplyr")
install.packages("SixSigma")
install.packages("ggplot2")
install.packages("mlapi") #A Sci-kit like API for Machine Learning models in R
library(dplyr)
library(SixSigma) 
library(ggplot2)
library(mlapi)
library(qcc)#To plot quality control charts
df = read.csv("E:\\Semester 3 Jun' 19 - Nov' 19\\R-Probability-Distributions_files\\MiningProcess_Flotation_Plant_Database.csv", dec = ",")
head(df)

#In the dataset we have to predict the % Silica Concentrate

#Silica Concentrate is the impurity in the iron ore which needs to be removed

#The current process of detecting silica takes many hours.

#With the help of some analysis and modelling of data we can give a good approximation of silica concentrate which will reduce a lot of 
#time and effort required for processing iron ore

#To find the number of observations recorded and the number of parameters considered in the data
dim(df)

#To remove all null values from the dataset and prepare it accordingly for analysis
df = na.omit(df)
dim(df)

#To find the details of the dataset as given to us such as mean, average and so on
summary(df)

#To convert the large dataframe into a numeric matrix which enables us to correlate each pairwise column of the dataframe for better analysis
df2 = as.matrix(as.data.frame(lapply(df, as.numeric)))

#To correlate all columns of the dataframe df pairwise
cor(df2)

#To plot a heatmap for the correlated matrix
heatmap(cor(df2))

#To remove unwanted vectors in the dataframe
dfs = within(df, rm('date', 'X..Iron.Concentrate', 'Ore.Pulp.pH', 'Flotation.Column.01.Air.Flow', 'Flotation.Column.02.Air.Flow', 'Flotation.Column.03.Air.Flow'))
head(dfs)

#To separate the variables
Y = df['X..Silica.Concentrate']
X = within(dfs, rm('X..Silica.Concentrate'))

X_scaled = fit_transform(X)


set.seed(12345)


#train_test_split: Split X and ydata into two portions according to input ratio. Default is for the split to include shuffling.
train_test_split <- function(X, y, test_size = 0.25, random_state = 0, shuffle = TRUE){
  # assure input types:
  if (!is.data.frame(X) & !is.atomic(X)) {
    stop('TypeError: X must be a dataframe or an atomic vector')}
  if (!is.data.frame(y) & !is.atomic(y)) {
    stop('TypeError: y must be a dataframe or an atomic vector')}
  if (!is.numeric(test_size)) {
    stop('TypeError: test_size must be a number')}
  if (!is.numeric(random_state)) {
    stop('TypeError: random_state must be a number')}
  if (!is.logical(shuffle)) {
    stop("TypeError: shuffle must be TRUE or FALSE")}
  
  # assure input values in range
  if (!(test_size>=0 & test_size<=1)) {
    stop('ValueError: test_size must be between 0 and 1')}
  if (!(random_state >= 0)) {
    stop('ValueError: random_state must be a nonnegative number')}
  
  
  # assure dimension match between X and y
  if (get_ncols(y)>1) {
    stop("DimensionError: y must not have more than one column")}
  if (get_nrows(X) != get_nrows(y)[1]) {
    stop("DimensionError: dimension of X does not equal dimension of y")}
  if (get_nrows(X) < 3) {
    stop("DimensionError: Sample size is less than 3, too small for splitting")}
  
  
  # Get splitting index Number
  N <- get_nrows(X)
  N_train <- round(N*(1-test_size))
  N_test <- N - N_train
  
  # Get indices
  if (shuffle == TRUE){
    set.seed(random_state)
    indice <- sample(N, N)
  }else{
    indice <- 1:N
  }
  
  # split X
  if (is.data.frame(X)){
    X_train <- X[indice[1:N_train],]
    X_test <- X[na.exclude(indice[N_train+1:N]),]
  }else{
    X_train <- X[indice[1:N_train]]
    X_test <- X[na.exclude(indice[N_train+1:N])]
  }
  # split y
  if (is.data.frame(y)){
    y_train <- y[indice[1:N_train],]
    y_test <- y[na.exclude(indice[N_train+1:N]),]
  }else{
    y_train <- y[indice[1:N_train]]
    y_test <- y[na.exclude(indice[N_train+1:N])]
  }
  
  # return results
  return(list(X_train = X_train, X_test = X_test,
              y_train = y_train, y_test = y_test))
}


get_nrows <- function(data){
  if (is.data.frame(data)){
    return(dim(data)[1])
  }else{
    return(length(data))
  }
}


get_ncols <- function(data){
  if (is.data.frame(data)){
    return(dim(data)[2])
  }else{
    return(1)
  }
}

#To use train_test_split to segregate data for analysis
split_train = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

X_train = as.data.frame(split_train["X_train"])
X_test = as.data.frame(split_train["X_test"])
y_train = as.data.frame(split_train["y_train"])
y_test = as.data.frame(split_train["y_test"])

#To build a linar regression model
mod1 = lm(X_train.X..Silica.Concentrate ~ X_train.X..Silica.Feed, data = X_train)

names(X_test)[names(X_test) == "X_test.X..Iron.Feed"] <- "X_train.X..Iron.Feed"
names(X_test)[names(X_test) == "X_test.X..Silica.Feed"] <- "X_train.X..Silica.Feed"
names(X_test)[names(X_test) == "X_test.Amina.Flow"] <- "X_train.Amina.Flow"
names(X_test)[names(X_test) == "X_test.Ore.Pulp.Flow"] <- "X_train.Ore.Pulp.Flow"
names(X_test)[names(X_test) == "X_test.Ore.Pulp.Density"] <- "X_train.Ore.Pulp.Density"
names(X_test)[names(X_test) == "X_test.Flotation.Column.04.Air.Flow"] <- "X_train.Flotation.Column.04.Air.Flow"
names(X_test)[names(X_test) == "X_test.Flotation.Column.05.Air.Flow"] <- "X_train.Flotation.Column.05.Air.Flow"
names(X_test)[names(X_test) == "X_test.Flotation.Column.06.Air.Flow"] <- "X_train.Flotation.Column.06.Air.Flow"
names(X_test)[names(X_test) == "X_test.Flotation.Column.07.Air.Flow"] <- "X_train.Flotation.Column.07.Air.Flow"
names(X_test)[names(X_test) == "X_test.Flotation.Column.01.Level"] <- "X_train.Flotation.Column.01.Level"
names(X_test)[names(X_test) == "X_test.Flotation.Column.02.Level"] <- "X_train.Flotation.Column.02.Level"
names(X_test)[names(X_test) == "X_test.Flotation.Column.03.Level"] <- "X_train.Flotation.Column.03.Level"
names(X_test)[names(X_test) == "X_test.Flotation.Column.04.Level"] <- "X_train.Flotation.Column.04.Level"
names(X_test)[names(X_test) == "X_test.Flotation.Column.05.Level"] <- "X_train.Flotation.Column.05.Level"
names(X_test)[names(X_test) == "X_test.Flotation.Column.06.Level"] <- "X_train.Flotation.Column.06.Level"
names(X_test)[names(X_test) == "X_test.Flotation.Column.07.Level"] <- "X_train.Flotation.Column.07.Level"
names(X_test)[names(X_test) == "X_test.X..Silica.Concentrate"] <- "X_train.X..Silica.Concentrate"
names(y_test)[names(y_test) == "y_test"] <- "X_train.X..Silica.Concentrate"
#To predict the values of the data based on the regression model constructed earlier
p1 = predict(mod1, newdata = X_test)



#Predicted values 
predict(mod1, newdata = y_test)

plot(p1)


#To check for process capability 
lcl = mean(df$Starch.Flow) - 6*sd(df$Starch.Flow)
ucl = mean(df$Starch.Flow) + 6*sd(df$Starch.Flow)

obs = select(df, c('X..Iron.Feed', 'X..Silica.Feed', 'Amina.Flow', 'Starch.Flow', 'Flotation.Column.04.Air.Flow'))

QCCGroups  <- function(data, sample){
  mat <- qcc::qcc.groups(data, sample)
  mat <- mat[which(apply(mat, 1, function(x) length(which(!is.na(x))))>1),,drop=FALSE]
  return(mat)
}

dat1 = obs$Starch.Flow[1:2000]
#Short term data analysis for the first 2000 samples

dat1l = obs$Starch.Flow[1:8000]
#Long term data correlation vector for the first 8000 samples

#To compute the process capability indices for starch flow
ss.ca.cp(dat1, LSL = lcl, USL = ucl, f.na.rm = TRUE)

#To create a Six Sigma capability analysis study for starch flow
ss.study.ca(dat1, LSL = lcl, USL = ucl, T = 4700.00, xLT = dat1l)

dat2 = obs$Amina.Flow[1:2000]

dat2l = obs$Amina.Flow[1:8000]
 
lcl1 = mean(df$Amina.Flow) - 6*sd(df$Amina.Flow)
ucl1 = mean(df$Amina.Flow) + 6*sd(df$Amina.Flow)

ss.ca.cp(dat2, LSL = lcl1, USL = ucl1, f.na.rm = TRUE)
ss.study.ca(dat2, LSL = lcl1, USL = ucl1, T = 650.00, xLT = dat2l)

#Six Sigma analysis for Ore Pulp Flow
dat3 = df$Ore.Pulp.Flow[1:2000]
dat3l = df$Ore.Pulp.Flow[1:8000]

lcl2 = mean(df$Ore.Pulp.Flow) - 3*sd(df$Ore.Pulp.Flow)
ucl2 = mean(df$Ore.Pulp.Flow) + 3*sd(df$Ore.Pulp.Flow)

ss.ca.cp(dat3, LSL = lcl2, USL = ucl2, f.na.rm = TRUE)
ss.study.ca(dat3, LSL = lcl2, USL = ucl2, T = 420.00, xLT = dat3l)

#Six Sigma Analysis for Flotation Column Levels
dat4 = df$Flotation.Column.05.Level[1:2000]
dat4l = df$Flotation.Column.05.Level[1:8000]

lcl3 = mean(df$Flotation.Column.05.Level) - 6*sd(df$Flotation.Column.05.Level)
ucl3 = mean(df$Flotation.Column.05.Level) + 6*sd(df$Flotation.Column.05.Level)

ss.ca.cp(dat4, LSL = lcl3, USL = ucl3, f.na.rm = TRUE)
ss.study.ca(dat4, LSL = lcl3, USL = ucl3, T = 520.00, xLT = dat4l)
