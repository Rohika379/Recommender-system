library(recommenderlab)
library(reshape2)

####### Example: Data generated in class #####
Game <-read.csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\Recomrndation engine\\game.csv",header=TRUE)
head(Game)


## covert to matrix format
?acast
game_matrix <- as.matrix(acast(Game, userId~game, fun.aggregate = mean))
dim(game_matrix)

## recommendarlab realRatingMatrix format
R <- as(game_matrix, "realRatingMatrix")


rec1 = Recommender(R, method="SVD")

rec2 = Recommender(R, method="POPULAR")


## create n recommendation
uid = "34"
Games_ <- subset(Game, Game$userId==uid)

print("You have rated:")
Games_

print("recommendations for you:")
prediction <- predict(rec1, R[uid], n=5) ## you may change the model here
as(prediction, "list")

prediction <- predict(rec2, R[uid], n=5) ## you may change the model here
as(prediction, "list")

