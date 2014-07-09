#from django.db import models
#from activeMDS.models import *
import os
import csv
from numpy import *
from numpy.random import *
import scipy.spatial

norm = linalg.norm
floor = math.floor
ceil = math.ceil

default_dir = '/'

def update_embedding(S,X,CV,start_iter=0,end_iter=nan):
    
    n = X.shape[0]
    d = X.shape[1]
    m = len(S)
    if isnan(end_iter):
        end_iter = 20*m
    
    count = 0
    avg_emp_loss = 0
    avg_hinge_loss = 0
    random_permutation = range(0,m)
    shuffle(random_permutation)
    for iter in range(start_iter,end_iter):
        #        G,avg_emp_loss,avg_hinge_loss = get_gradient(X,S)
        #        eta = 400.
        
        q = S[random_permutation[count]]
        G,emp_loss,hinge_loss = get_gradient(X,[q])
        avg_emp_loss = avg_emp_loss + emp_loss
        avg_hinge_loss = avg_hinge_loss + hinge_loss
        count = count + 1
        eta = float(sqrt(100))/sqrt(iter+100)
        
        X = X - eta*G
        if iter % m == 0:
            MDS = X/norm(X)*sqrt(n)
            print test_cv(MDS,CV)   
            print "epoch = "+str(iter/m)+"   emp_loss = "+str(avg_emp_loss/count)+"   hinge_loss = "+str(avg_hinge_loss/count)+"    norm(X)/sqrt(n) = "+str(norm(X)/sqrt(n))
            avg_emp_loss = 0
            avg_hinge_loss = 0
            count = 0
            shuffle(random_permutation)

    return X/norm(X)*sqrt(n)

def get_gradient(X,S):
    # returns gradient wrt loss function 1/m sum_{ell = 1}^m loss(X,S[ell,:])
    
    n = X.shape[0]
    d = X.shape[1]
    m = len(S)*1.
    
    emp_loss = 0 # 0/1 loss
    hinge_loss = 0 # hinge loss
    
    # S[iter,:] = [i,j,k]   <=>    norm(xi-xk)<norm(xj-xk) )
    H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
    
    G = zeros((n,d))
    for q in S:
        loss_ijk = trace(dot(H,dot(X[q,:],X[q,:].T)))
        if loss_ijk+1.>0.:
            hinge_loss = hinge_loss + loss_ijk + 1.
            G[q,:] = G[q,:] + H*X[q,:]/m
            
            if loss_ijk > 0:
                emp_loss = emp_loss + 1.
    

    emp_loss = emp_loss/m
    hinge_loss = hinge_loss/m
    
    return G, emp_loss, hinge_loss
    
def test_cv(MDS,CV):
    n = len(CV)
    cv_err = 0
    for i in xrange(n):
        xy = [MDS[j] for j in CV[i]]
        d = scipy.spatial.distance.pdist(xy)
        
        if (d[0] < d[1]):
            cv_err = cv_err + 1
    return(cv_err/float(n))
    