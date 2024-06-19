library(mboost)

set.seed(2010)

data_amount <- 500

x1 <- rnorm(data_amount)
x2 <- rnorm(data_amount)
x3 <- rnorm(data_amount)
x4 <- runif(data_amount)
x5 <- runif(data_amount)

#Split
train <- data_amount*0.8
test <- data_amount - train

#Boosting parameters
flood_level <- 0
learning_rate = 0.01
rounds <- 500

#Target values
y <- 0 + x1 + x2 + x3 + x4 + x5

y_training <- y[1:train]

#Adding training data to a dataframe
dat <- data.frame(y_training,x1[1:train], x2[1:train], x3[1:train], x4[1:train], x5[1:train])


ctrl <- boost_control(mstop = rounds, ## number of boosting iterations
                      nu = learning_rate, ## step length
                      trace = TRUE)

#Creating a Family with the flooded L2loss and gradient descent and ascent
l2loss_flooding <- function(){ 
   Family( ## applying the Family function
     loss = function(y, f) abs(((y - f)^2) - flood_level)+flood_level,
     ngradient = function(y, f, w = 1)
       if(((mean((y - f)^2)) - flood_level)>= 0){ 
         y - f
       }else{
         -(y - f)
       },
     offset = weighted.mean,
     name = "Family for L2 loss with flooding" )}

glm1 <- glmboost(y_training ~ ., data = dat, family = l2loss_flooding(), control = ctrl)

plot(glm1)


training_preds <- double(train)
testing_preds <- double(test)
training_loss <- double(rounds)
testing_loss <- double(rounds)

for (j in 1:rounds){
  coefficients <- coef(glm1[j], which = "")
  
  #Calculates all the models training set predictions for the j-th round
  for (i in 1:train){
    x_train <- c(1, x1[i], x2[i], x3[i], x4[i], x5[i])
    training_preds[i] <- sum(x_train %*% coefficients)
  }
  #Calculates all the models testing set predictions for the j-th round
  for (i in (train+1):data_amount){
    x_test <- c(1, x1[i], x2[i], x3[i], x4[i], x5[i])
    testing_preds[(i-train)] <- sum(x_test %*% coefficients)
  }

  #Calculates the MSE of the j-th round
  training_loss[j] <- (sum((y_training - training_preds)^2)) / (train)
  testing_loss[j] <- (sum((y[(train+1):length(y)] - testing_preds)^2)) /(data_amount-train)
}

matplot(1:rounds, cbind(training_loss, testing_loss), type = "l", lty = 1,
        col = c("blue", "red"), xlab = "Iterations", 
        ylab = "Loss", main = "",cex.lab=1.4, cex.axis=1.4, cex.main=1.4, cex.sub=1.4)
legend("topright", legend = c("Training loss", "Testing loss"), 
       col = c("blue", "red"), cex = 1,
       lty = 1)
