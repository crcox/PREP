library(dplyr)
makeDict <- function(keys,values){
  dict <- list()
  for (i in seq(1:length(keys))) {
    mapping[[ keys[i] ]] <- values[i]
  }
}

data <- read.csv('Animal Similarity Comparison-export-Thu Jun 26 16-27-40 CDT 2014.csv')
mapping <- read.csv('discovery animals 2.csv',header=FALSE)

summary(data)

Q <- data[,c('target','alternate','primary')] 
S <- as.matrix(Q)
head(Q)

n <- max(n_distinct(data$target),n_distinct(data$alternate),n_distinct(data$primary)
d <- 5

X <- matrix(rnorm(n*d),nrow=n,ncol=d)
X <- X/norm(X,type='F')*sqrt(n)

MDS = update_embedding(S,X,0,length(S)*100)

