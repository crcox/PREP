makeDict <- function(keys,values){
  dict <- list()
  for (i in seq(1:length(keys))) {
    mapping[[ keys[i] ]] <- values[i]
  }
}

d <- read.csv('Animal Similarity Comparison-export-Thu Jun 26 16-27-40 CDT 2014.csv')
temp <- read.csv('discovery animals 2.csv',header=FALSE)



