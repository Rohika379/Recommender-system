
library(recommenderlab)
library(reshape2)

####### Example: Data generated in class #####
ratings_list <-read.csv("C:\\Users\\rohika\\OneDrive\\Desktop\\360digiTMG assignment\\Recomrndation engine\\Entertainment.csv",header=TRUE)
head(ratings_list)

## covert to matrix format
?acast
ratings_matrix <- as.matrix(acast(ratings_list, ï..Id ~Titles, fun.aggregate = mean))
dim(ratings_matrix)

## recommendarlab realRatingMatrix format
R <- as(ratings_matrix, "realRatingMatrix")


rec1 = Recommender(R, method="POPULAR")


## create n recommendations for a user
uid = "5355"
ent <- subset(ratings_list, ratings_list$ï..Id==uid)
print("You have rated:")
ent
print("recommendations for you:")

prediction <- predict(rec1, R[uid], n=99) ## you may change the model here
as(prediction, "list")

