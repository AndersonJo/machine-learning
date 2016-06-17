train.data.raw <- read.csv('machine-error01.csv', header = T)
train.data.raw <- subset(train.data.raw, select = -c(2))

data <- train.data.raw
train <- data[1:13000,]
test <- data[13001:14375,]

model = glm(formula=error~., family=binomial(link='logit'), data=train)

fitted.results <- predict(model, newdata=test, type='response')
fitted.results <- ifelse(fitted.results > 0.5, 1, 0)
mean(fitted.results == test$error)
