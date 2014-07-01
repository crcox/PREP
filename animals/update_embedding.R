update_embedding <- function(S,X,start_iter=0,end_iter=NA) {
  require(dplyr)
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
  
  pp <- Progress(end_iter)
  for (iter in seq(start_iter,end_iter)) {
    pp$tick()$show()
    count = count + 1
    q = S[random_permutation[count],]
    W = get_gradient(X,q)
    avg_emp_loss = avg_emp_loss + W$emp_loss
    avg_hinge_loss = avg_hinge_loss + W$hinge_loss
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