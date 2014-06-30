library(dplyr)
makeDict <- function(keys,values){
  dict <- list()
  for (i in seq(1:length(keys))) {
    mapping[[ keys[i] ]] <- values[i]
  }
}

update_embedding <- function(S,X,start_iter=0,end_iter=NA) {
  n = nrow(X)
  d = ncol(X)
  m = nrow(S)
  
  if (is.na(end_iter)) {
    end_iter = 20*m
  }

  count = 0
  avg_emp_loss = 0
  avg_hinge_loss = 0
  random_permutation = base::sample(m)
  
  for (iter in seq(start_iter,end_iter)) {
    q = S[random_permutation[count],]
    W = get_gradient(X,q)
    avg_emp_loss = avg_emp_loss + W$emp_loss
    avg_hinge_loss = avg_hinge_loss + W$hinge_loss
    count = count + 1
    eta = (sqrt(100))/sqrt(iter+100)
    X = X - eta*W$G
    
    if (iter %% m == 0) {
      #            print "epoch = "+str(iter/m)+"   emp_loss = "+str(avg_emp_loss/count)+"   hinge_loss = "+str(avg_hinge_loss/count)+"    norm(X)/sqrt(n) = "+str(norm(X)/sqrt(n))
      avg_emp_loss = 0
      avg_hinge_loss = 0
      count = 0
      random_permutation = base::sample(m)
    } 
  }
  return(X/norm(X,type='F')*sqrt(n))
}

get_gradient <- function(X,S) {
  # returns gradient wrt loss function 1/m sum_{ell = 1}^m loss(X,S[ell,:])
  n = nrow(X)
  d = ncol(X)
  m = nrow(S)
  
  emp_loss = 0 # 0/1 loss
  hinge_loss = 0 # hinge loss
  
  # S[iter,:] = [i,j,k]   <=>    norm(xi-xk)<norm(xj-xk) )
  H = matrix(c(1,0,-1,0,-1,1,-1,1,0), nrow=3, ncol=3)
  
  G = matrix(0,nrow=n,ncol=d)
  for (q in S) {
    loss_ijk = sum(diag((H %*% (X[q,] %*% t(X[q,])))))
    
    if (loss_ijk+1 > 0) {
      hinge_loss = hinge_loss + loss_ijk + 1
      G[q,] = G[q,] + H*X[q,]/m
      
      if (loss_ijk > 0) {
        emp_loss = emp_loss + 1
      }
    }
  }
    
  emp_loss = emp_loss/m
  hinge_loss = hinge_loss/m

  return(list(G=G, emp_loss=emp_loss, hinge_loss=hinge_loss))
}



data <- read.csv('Animal Similarity Comparison-export-Thu Jun 26 16-27-40 CDT 2014.csv')
mapping <- read.csv('discovery animals 2.csv',header=FALSE)

summary(data)

Q <- data[,c('target','alternate','primary')] - 1 
S <- as.matrix(Q)
head(Q)

n <- n_distinct(data$target)
d <- 5

X <- matrix(rnorm(n*d),nrow=n,ncol=d)
X <- X/norm(X,type='F')*sqrt(n)

MDS = update_embedding(S,X,0,length(S)*100)

