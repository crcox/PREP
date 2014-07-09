import os
import csv
from numpy import *
from numpy.random import *
import activeMDS

reload(activeMDS)

norm = linalg.norm
floor = math.floor
ceil = math.ceil

with open("C:/Users/chris/PREP/animals/Animal Similarity Comparison-export-Thu Jun 26 16-27-40 CDT 2014.csv", "r") as f:
    headings = f.readline().strip().split(',')
    qtype_ind = headings.index('queryType')
    d = { k:[] for k in headings }
    cv = { k:[] for k in headings }
    for line in f:
        data = line.strip().split(',')
        if int(data[qtype_ind]) == 1:
            [d[k].append(v) for k,v in zip(headings,data)]
        else:
            #[d[k].append(v) for k,v in zip(headings,data)]
            [cv[k].append(v) for k,v in zip(headings,data)]

d['primary'] = [int(x) for x in d['primary']]
d['alternate'] = [int(x) for x in d['alternate']]
d['target'] = [int(x) for x in d['target']]

mp = min(d['primary'])
ma = min(d['alternate'])
mt = min(d['target'])
mm = min([mp,ma,mt])
S = [ [int(p)-mm,int(a)-mm,int(t)-mm] for p,a,t in zip(d['primary'],d['alternate'],d['target']) ]
CV = [ [int(p)-mm,int(a)-mm,int(t)-mm] for p,a,t in zip(cv['primary'],cv['alternate'],cv['target']) ]

n = max(max(S)+max(CV)) + 1
d = 2

X = randn(n,d)
X = X/norm(X)*sqrt(n)

X = activeMDS.update_embedding(S,X,CV,0,len(S)*100)
##
##print X
