import os
import csv
from numpy import *
from numpy.random import *
import activeMDS

norm = linalg.norm
floor = math.floor
ceil = math.ceil

with open("Animal Similarity Comparison-export-Thu Jun 26 16-27-40 CDT 2014.csv", "r") as f:
	headings = f.readline().strip().split(',')
	d = { k:[] for k in headings }
	for line in f:
		data = line.strip().split(',')
		[d[k].append(v) for k,v in zip(headings,data)]

d['primary'] = [int(x) for x in d['primary']]
d['alternate'] = [int(x) for x in d['alternate']]
d['target'] = [int(x) for x in d['target']]

mp = min(d['primary'])
ma = min(d['alternate'])
mt = min(d['target'])
S = [ [int(p)-mp,int(a)-ma,int(t)-mt] for p,a,t in zip(d['primary'],d['alternate'],d['target']) ]

n = len(set(d['primary']))
d = 5

X = randn(n,d)
X = X/norm(X)*sqrt(n)

print n,d
X = activeMDS.update_embedding(S,X,0,len(S)*100)

print X
