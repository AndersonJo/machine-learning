train.data.raw <- read.csv('machine-error02.csv', header = T)
train.data.raw <- subset(train.data.raw, select = c(1, 3, 4, 5, 6, 7, 8))

data <- train.data.raw
train <- data[1:14000,]
test <- data[14001:14375,]

model = glm(formula=error~., family=binomial(link='logit'), data=train)

fitted.results <- predict(model, newdata=test, type='response')
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)
mean(fitted.results == test$error)
