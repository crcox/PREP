get_gradient <- function(X,q) {
  # returns gradient wrt loss function 1/m sum_{ell = 1}^m loss(X,S[ell,:])
  n = nrow(X)
  d = ncol(X)
  emp_loss = 0 # 0/1 loss
  hinge_loss = 0 # hinge loss
  
  # S[iter,:] = [i,j,k]   <=>    norm(xi-xk)<norm(xj-xk) )
  H = matrix(c(1,0,-1,0,-1,1,-1,1,0), nrow=3, ncol=3)
  
  G = matrix(0,nrow=n,ncol=d)
  loss_ijk = sum(diag((H %*% (X[q,] %*% t(X[q,])))))
  
  if (loss_ijk+1 > 0) {
    hinge_loss = hinge_loss + loss_ijk + 1
    G[q,] = G[q,] + (H%*%X[q,])
    
    if (loss_ijk > 0) {
      emp_loss = emp_loss + 1
    }
  }

  
  emp_loss = emp_loss
  hinge_loss = hinge_loss
  
  return(list(G=G, emp_loss=emp_loss, hinge_loss=hinge_loss))
}