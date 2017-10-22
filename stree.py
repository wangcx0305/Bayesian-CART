# -*- coding: utf-8 -*-
import pandas as pa
import numpy as np
from numpy import log
import random as rn
from math import sin, cos, sqrt, atan2, radians
from scipy.linalg import block_diag, det, inv
from scipy.sparse.linalg import cg
from scipy.stats import multivariate_normal, uniform, poisson
from numpy.random import choice, gamma
from copy import deepcopy
datainitial = pa.read_csv('zinc_mine.csv')
data = datainitial[datainitial.Lichen_Cover > 0]
datanorth = data[data.Lat > 67.4 ]
datanorth = datanorth.reset_index()
datanorth["Lichen_Cover_Logit"] = np.log(datanorth.Lichen_Cover/(100 - datanorth.Lichen_Cover))
indvar = datanorth.ix[:, 5:11]
n = len(datanorth)
varlist = data.columns.values[4:9]

class Node:
    def __init__(self, var, val, left = None, right = None, parent = None):
        self.var = var
        self.val = val
        self.leftchild = left
        self.rightchild = right
        self.parent = parent
        
    def prin(self):
       if(self.leftchild):
           self.leftchild.prin()
       print(self.var, self.val)
       if(self.rightchild):
           self.rightchild.prin()
    
    def isleaf(self):
        return self and not (self.leftchild or self.rightchild)
          
    def growable(self):
        return self and not (self.leftchild and self.rightchild)
        
    def growablelist(self, a):
        if(self.leftchild):
           self.leftchild.growablelist(a)
        if(self.rightchild):
           self.rightchild.growablelist(a)
        if(self.growable()):
            if(self.isleaf()):
               a.append(self)
               a.append(self)
            else:
                a.append(self)
        return a
        
    def grow(self, var, val):
        if(self.growable()):
            if(self.isleaf()):
               if(bool(rn.getrandbits(1))):
                 self.leftchild = Node(var, val, parent = self)
               else:
                 self.rightchild = Node(var, val, parent = self)
            elif ((self.leftchild) and (not self.rightchild)):
                self.rightchild = Node(var, val, parent = self)
            else:
                self.leftchild = Node(var, val, parent = self)
                
    def prunable(self):
        return self.isleaf()
        
    def prunablelist(self,a):
        if(self.leftchild):
           self.leftchild.prunablelist(a)
        if(self.rightchild):
           self.rightchild.prunablelist(a)
        if(self.prunable()):
            a.append(self)
        return a
        
    def isleftchild(self):
        return self.parent and self.parent.leftchild is self
        
    def isrightchild(self):
        return self.parent and self.parent.rightchild is self
        
    def prune(self):
        if(self.prunable()):
            if (self.isleftchild()):
                self.parent.leftchild = None
            else:
                self.parent.rightchild = None
                              
    def largestleftparent(self, var):
        flag = -1
        while self.parent:
            if (self.parent.rightchild is self and self.parent.var is var):
                if(self.parent.val > flag):
                    flag = self.parent.val
            self = self.parent
        return flag
        
    def smallestrightparent(self,var):
        flag = 100000
        while self.parent:
            if (self.parent.leftchild is self and self.parent.var is var):
                if(self.parent.val < flag):
                    flag = self.parent.val
            self = self.parent
        return flag
             
    def largestvarsubtree(self, var):
        flag = -1
        if(self.leftchild):
            self.leftchild.largestvarsubtree(var)
        if(self.rightchild):
            self.rightchild.largestvarsubtree(var)
        if(self and (self.var is var)):
            if(flag < self.val):
               flag = self.val 
        return flag
            
    def smallestvarsubtree(self,var):
        flag = 1000000
        if(self.leftchild):
            self.leftchild.smallestvarsubtree(var)
        if(self.rightchild):
            self.rightchild.smallestvarsubtree(var)
        if(self and (self.var is var)):
            if(flag > self.val):
                flag = self.val
        return flag
        
    def perturblist(self, a):
        if(self.leftchild):
            self.leftchild.perturblist(a)
        if(self.rightchild):
            self.rightchild.perturblist(a)
        if(self):
            a.append(self)
        return a
        
    def perturbrange(self, var):
        self.var = var
        largestleftparent = -1
        if(self.parent):
            largestleftparent = self.largestleftparent(var)
        smallestrightparent = 100000
        if(self.parent):
            smallestrightparent = self.smallestrightparent(var)
        largestleftsubtree = -1
        if(self.leftchild):
           largestleftsubtree = self.leftchild.largestvarsubtree(var)
        smallestrightsubtree = 100000
        if(self.rightchild):
           smallestrightsubtree = self.rightchild.smallestvarsubtree(var)
        lmin = max(min(datanorth[var]), largestleftparent, largestleftsubtree)
        lmax = min(max(datanorth[var]), smallestrightparent, smallestrightsubtree)
        return [lmin, lmax]
        
    def onenodedata(self, data):
        if(self.leftchild and self.rightchild):
            print("don't have terminal")
        elif(not self.leftchild and self.rightchild):
            c = data[data[self.var] < self.val]
            while(self.parent):
                if(self.isleftchild()):
                    c = c[c[self.parent.var] < self.parent.val]
                else:
                    c = c[c[self.parent.var] >= self.parent.val]
                self = self.parent
            return c              
        elif(self.leftchild and not self.rightchild):
            c = data[data[self.var] >= self.val]
            while(self.parent):
                if(self.isleftchild()):
                    c = c[c[self.parent.var] < self.parent.val]
                else:
                    c = c[c[self.parent.var] >= self.parent.val]
                self = self.parent
            return c
        else:
            a = data[data[self.var] >= self.val]
            b = data[data[self.var] < self.val]
            while(self.parent):
                if(self.isleftchild()):
                    a = a[a[self.parent.var] < self.parent.val]
                    b = b[b[self.parent.var] < self.parent.val]
                else:
                    a = a[a[self.parent.var] >= self.parent.val]
                    b = b[b[self.parent.var] >= self.parent.val]
                self = self.parent
            return [a, b]
        
    def allnodedata(self, data, a):
         if(self.leftchild):
             a = self.leftchild.allnodedata(data, a)
         if(self.rightchild):
             a = self.rightchild.allnodedata(data, a)
         if(self.growable()):
             if(self.rightchild and not self.leftchild):
                 b = self.onenodedata(data)
                 a.append(b)
             elif(self.leftchild and not self.rightchild):
                 b = self.onenodedata(data)
                 a.append(b)
             else:
                 [c, d] = self.onenodedata(data)
                 a.append(c)
                 a.append(d)
         return a
                           
class Tree:
    def __init__(self):
        self.root = None
        self.size = 0
        
    def grow(self, var, val):
        l = list()
        a = self.root.growablelist(l)
        if(not a):
            print("Tree is not growable")
        else:
            b = rn.choice(a)
            b.grow(var, val)
            self.size += 1
            
    def prune(self):
        l = list()
        a = self.root.prunablelist(l)
        if(not a):
            print("Tree is not prunable")
        else:
            b = rn.choice(a)
            b.prune()
            self.size -= 1
            
    def prin(self):
        if (not self.root):
            print("empty tree")
        else:
            self.root.prin()
            
    def treedata(self, data):
        if(not self.root):
            print("empty tree")
        else:
            l = []
            l = self.root.allnodedata(data,l)
            return l
            
    def number_check(self, data):
        if(not self.root):
            print("empty tree")
        else:
            l = self.treedata(data)
            for i in range(len(l)):
                if (len(l[i]) < 5):
                    return 0
            return 1
        
    def death_number(self):
        l = list()
        a = self.root.prunablelist(l)
        return len(a)
    
            
    
def dist(lat1, lon1, lat2, lon2):
    R = 6378.1
    ralat1 = radians(lat1)
    ralon1 = radians(lon1)
    ralat2 = radians(lat2)
    ralon2 = radians(lon2)  
    dlon = ralon2 - ralon1
    dlat = ralat2 - ralat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance
      
def pairexpcovkernel(data, i, j, rpa):
    lat1 = data.Lat[i]
    lon1 = data.Lon[i]
    lat2 = data.Lat[j]
    lon2 = data.Lon[j]
    return np.exp(-dist(lat1, lon1,lat2, lon2)/rpa)     
        
def covkernel(data, lambda0): 
     dim = len(data.index)
     cov = [[0 for x in range(dim)] for y in range(dim)]
     for i in range(0, dim):
         for j in range(0, dim):
             cov[i][j] = pairexpcovkernel(data, i, j, lambda0)
     return cov


rn.seed(305)
var = rn.choice(varlist)   
val = rn.choice(datanorth[var])
mytree = Tree()
mytree.root = Node(var, val)
mytree.size = 1
nodedata = mytree.treedata(datanorth)
index = list()
for i in range(len(nodedata)):
  index = np.append(index, nodedata[i].index.values)
 
R = np.matrix(np.zeros(shape = (len(index), len(index))))
for i in range(len(index)):
  R[i, index[i]] = 1
 
x = list()
for i in range(len(nodedata)):
  x.append(np.matrix(indvar[indvar.index.isin(nodedata[i].index.values)].values))
x = np.matrix(block_diag(x[0],x[1]))
   
y = list()
for i in range(len(nodedata)):
  y = np.append(y, nodedata[i].Lichen_Cover_Logit.values)
y = np.matrix(y).T    
 #initial values
lambda0 = 1.0
lambda1 = 1.0
lambda2 = 1.0
V = covkernel(datanorth, lambda0)
iV = inv(V)
psi = np.matrix(sqrt(1.0 / lambda1) * multivariate_normal.rvs(np.zeros(91), V)).T
 
qbeta = (lambda2 + 1) * (x.T * x)
mubeta = np.matrix(cg(qbeta, lambda2 * x.T * (y - R * psi))[0]).T
betatemp = np.matrix(multivariate_normal.rvs(np.zeros(len(mubeta)), qbeta)).T
beta = mubeta + np.matrix(cg(qbeta, betatemp)[0]).T
 
 
qpsi = lambda2 * (R.T * R) + lambda1 * iV
mupsi =  np.matrix(cg(qpsi, lambda2 * R.T * (y - x * beta))[0]).T
psitemp = np.matrix(multivariate_normal.rvs(np.zeros(len(mupsi)), qpsi)).T
psi = mupsi + np.matrix(cg(qpsi, mupsi)[0]).T
 
lambda2 = gamma(len(datanorth) / 2 + 1, 
          1 / ((y - x * beta - R * psi).T * (y - x * beta - R * psi) + 1 / 200))

lambda1 = gamma(len(datanorth) / 2 + 1, 1 / (psi.T * iV * psi + 1 / 200)) 

lambda0temp = uniform.rvs(0.01, 100)    
Vtemp = covkernel(datanorth, lambda0temp)  
iVtemp = inv(Vtemp)    
alpha = -0.5 * log(det(Vtemp)) - lambda1 * psi.T * iVtemp * psi + log(lambda0 -0.01) \
        + 0.5 * log(det(V)) + lambda1 * psi.T * iV * psi - log(lambda0temp - 0.01)   
alpha = min(0 ,alpha)      
u = uniform.rvs(0, 1) 
if(alpha > log(u)):
 lambda0 = lambda0temp
 V = deepcopy(Vtemp)
 iV = deepcopy(iVtemp)
 
###set the rate parameter for Poisson distribution as 10
lam = 10
treelist = list()
treesizelist = list()
lambda0list = list()
lambda1list = list()
lambda2list = list()
vpsdlist = list()
treelist.append(mytree)
treesizelist.append(mytree.size)
lambda0list.append(lambda0)
lambda1list.append(lambda1)
lambda2list.append(lambda2)
vpsdlist.append(np.all(np.linalg.eigvals(V) >= 0))

 
 
for i in range(50000):
    protree = deepcopy(mytree)
    k = mytree.size + 1
    if(k <= 0):
        print('error')
    elif (k <= 2):
        var = rn.choice(varlist)
        val = rn.choice(datanorth[var])
        protree.grow(var,val)
        if (protree.number_check(datanorth) == 1):
            mytree = deepcopy(protree)
            nodedata = mytree.treedata(datanorth)
            index = list()
            for i in range(len(nodedata)):
                index = np.append(index, nodedata[i].index.values)
            R = np.matrix(np.zeros(shape = (len(index), len(index))))
            for i in range(len(index)):
                R[i, index[i]] = 1
            xlist = list()
            for i in range(len(nodedata)):
                xlist.append(np.matrix(indvar[indvar.index.isin(nodedata[i].index.values)].values))
            temp = list()
            for i in range(len(nodedata)):
                  temp = np.matrix(block_diag(temp,xlist[i]))
            x = deepcopy(temp[1:92, :])
            y = list()
            for i in range(len(nodedata)):
                y = np.append(y, nodedata[i].Lichen_Cover_Logit.values)
            y = np.matrix(y).T 
            
            qbeta = (lambda2 + 1) * (x.T * x)
            mubeta = np.matrix(cg(qbeta, lambda2 * x.T * (y - R * psi))[0]).T
            betatemp = np.matrix(multivariate_normal.rvs(np.zeros(len(mubeta)), qbeta)).T
            beta = mubeta + np.matrix(cg(qbeta, betatemp)[0]).T
 
 
            qpsi = lambda2 * (R.T * R) + lambda1 * iV
            mupsi =  np.matrix(cg(qpsi, lambda2 * R.T * (y - x * beta))[0]).T
            psitemp = np.matrix(multivariate_normal.rvs(np.zeros(len(mupsi)), qpsi)).T
            psi = mupsi + np.matrix(cg(qpsi, mupsi)[0]).T
 
            lambda2 = gamma(len(datanorth) / 2 + 1, 
                      1 / ((y - x * beta - R * psi).T * (y - x * beta - R * psi) + 1 / 200))

            lambda1 = gamma(len(datanorth) / 2 + 1, 1 / (psi.T * iV * psi + 1 / 200)) 

            lambda0temp = uniform.rvs(0.01, 100)    
            Vtemp = covkernel(datanorth, lambda0temp)  
            iVtemp = inv(Vtemp)    
            alpha = -0.5 * log(det(Vtemp)) - lambda1 * psi.T * iVtemp * psi + log(lambda0 -0.01) \
                     + 0.5 * log(det(V)) + lambda1 * psi.T * iV * psi - log(lambda0temp - 0.01)   
            alpha = min(0 ,alpha)      
            u = uniform.rvs(0, 1) 
            if(alpha > log(u)):
               lambda0 = lambda0temp
               V = deepcopy(Vtemp)
               iV = deepcopy(iVtemp)
    else:
        c = 0.75/2
        b = min(1, poisson.pmf(k + 1, lam)/poisson.pmf(k, lam))
        d = min(1, poisson.pmf(k - 1, lam)/poisson.pmf(k, lam))
        bp = b * c
        dp = d * c
        cnu = crho = (1 - bp - dp) / 2
        u = uniform.rvs(0,1)
        if(u <= bp):
           var = choice(varlist)
           val = choice(datanorth[var])
           protree.grow(var, val)
           if(protree.number_check(datanorth) == 0):
              pass
           else:
              pronodedata = protree.treedata(datanorth)
              proindex = list()
              for i in range(len(pronodedata)):
                 proindex = np.append(proindex, pronodedata[i].index.values)
         
              proR = np.matrix(np.zeros(shape = (len(proindex), len(proindex))))
              for i in range(len(proindex)):
                 proR[i, proindex[i]] = 1
         
              proxlist = list()
              for i in range(len(pronodedata)):
                 proxlist.append(np.matrix(\
                    indvar[indvar.index.isin(pronodedata[i].index.values)].values))
              temp = list()
              for i in range(len(pronodedata)):
                  temp = np.matrix(block_diag(temp,proxlist[i]))
              prox = deepcopy(temp[1:92, :])
              
              proy = list()
              for i in range(len(pronodedata)):
                 proy = np.append(proy, pronodedata[i].Lichen_Cover_Logit.values)
              proy = np.matrix(proy).T
              
              proqbeta = (lambda2 + 1) * (prox.T * prox)
              promubeta = np.matrix(cg(proqbeta, \
                             lambda2 * prox.T * (proy - proR * psi))[0]).T
              probetatemp = np.matrix(multivariate_normal.rvs(np.zeros(len(promubeta)), \
                                     proqbeta)).T
              probeta = promubeta + np.matrix(cg(proqbeta, probetatemp)[0]).T
     
     
              proqpsi = lambda2 * (proR.T * proR) + lambda1 * iV
              promupsi =  np.matrix(cg(proqpsi, \
                            lambda2 * proR.T * (proy - prox * probeta))[0]).T
              propsitemp = np.matrix(multivariate_normal.rvs(np.zeros(len(promupsi)), proqpsi)).T
              propsi = promupsi + np.matrix(cg(proqpsi, promupsi)[0]).T
     
              prolambda2 = gamma(len(datanorth) / 2 + 1, 
                          1 / ((proy - prox * probeta - proR * propsi).T * (proy - prox * probeta - proR * propsi) + 1 / 200))
        
              prolambda1 = gamma(len(datanorth) / 2 + 1, 1 / (propsi.T * iV * propsi + 1 / 200)) 
    
    
              prolambda0 = deepcopy(lambda0)
              proV = deepcopy(V)
              proiV = deepcopy(iV)
              prolambda0temp = uniform.rvs(0.01, 100)
              proVtemp = covkernel(datanorth, prolambda0temp)  
              proiVtemp = inv(proVtemp)    
              alpha = -0.5 * log(det(proVtemp)) - prolambda1 * propsi.T * proiVtemp * propsi + log(prolambda0 -0.01) \
                         + 0.5 * log(det(proV)) + prolambda1 * propsi.T * proiV * propsi - log(prolambda0temp - 0.01)   
              alpha = min(0 ,alpha)      
              u = uniform.rvs(0, 1) 
              if(alpha > log(u)):
                 prolambda0 = deepcopy(prolambda0temp)
                 proV = deepcopy(Vtemp)
                 proiV = deepcopy(iVtemp)
           
              kdie = mytree.death_number()
              
              lpropl =  (n/2) * log(prolambda2) - prolambda2 * (proy - prox * probeta - proR * propsi).T * (proy - prox * probeta - proR * propsi) + \
                        (n/2) * log(prolambda1) - 0.5 * log(det(proV)) - prolambda1 * propsi.T * proiV * propsi + \
                        0.5 * log(det(prox.T * prox)) - probeta.T * (prox.T * prox) * probeta - prolambda2 / 200 - \
                        prolambda1 / 200 + log(prolambda0 - 0.01) + log(k)
              lcurpl = (n/2) * log(lambda2) - lambda2 * (y - x * beta - R * psi).T * (y - x * beta - R * psi) + \
                        (n/2) * log(lambda1) - 0.5 * log(det(V)) - lambda1 * psi.T * iV * psi + \
                         0.5 * log(det(x.T * x)) - beta.T * (x.T * x) * beta - lambda2 / 200 - \
                          lambda1 / 200 + log(lambda0 - 0.01) + log(kdie + 1)
           
              alpha = min(0, lpropl - lcurpl)
              u = uniform.rvs(0, 1)
              if(alpha > log(u)):
                 mytree = deepcopy(protree)
                 lambda0 = deepcopy(prolambda0)
                 lambda1 = deepcopy(prolambda1)
                 lambda2 = deepcopy(prolambda2)
                 V = deepcopy(proV)
                 iV = deepcopy(proiV)
                 beta = deepcopy(probeta)
                 psi = deepcopy(propsi)
                 R = deepcopy(proR)
                 x = deepcopy(prox)
                 y = deepcopy(proy)      
        elif(bp < u <= bp + dp):
             #mytree = deepcopy(protree)
             protree.prune()
             if(protree.number_check(datanorth) == 0):
                 pass 
             else:  
              pronodedata = protree.treedata(datanorth)
              proindex = list()
              for i in range(len(pronodedata)):
                 proindex = np.append(proindex, pronodedata[i].index.values)
         
              proR = np.matrix(np.zeros(shape = (len(proindex), len(proindex))))
              for i in range(len(proindex)):
                 proR[i, proindex[i]] = 1
         
              proxlist = list()
              for i in range(len(pronodedata)):
                 proxlist.append(np.matrix(\
                    indvar[indvar.index.isin(pronodedata[i].index.values)].values))
              temp = list()
              for i in range(len(pronodedata)):
                  temp = np.matrix(block_diag(temp,proxlist[i]))
              prox = deepcopy(temp[1:92, :])
              
              proy = list()
              for i in range(len(pronodedata)):
                 proy = np.append(proy, pronodedata[i].Lichen_Cover_Logit.values)
              proy = np.matrix(proy).T
              
              proqbeta = (lambda2 + 1) * (prox.T * prox)
              promubeta = np.matrix(cg(proqbeta, \
                             lambda2 * prox.T * (proy - proR * psi))[0]).T
              probetatemp = np.matrix(multivariate_normal.rvs(np.zeros(len(promubeta)), \
                                     proqbeta)).T
              probeta = promubeta + np.matrix(cg(proqbeta, probetatemp)[0]).T
     
     
              proqpsi = lambda2 * (proR.T * proR) + lambda1 * iV
              promupsi =  np.matrix(cg(proqpsi, \
                            lambda2 * proR.T * (proy - prox * probeta))[0]).T
              propsitemp = np.matrix(multivariate_normal.rvs(np.zeros(len(promupsi)), proqpsi)).T
              propsi = promupsi + np.matrix(cg(proqpsi, promupsi)[0]).T
     
              prolambda2 = gamma(len(datanorth) / 2 + 1, 
                          1 / ((proy - prox * probeta - proR * propsi).T * (proy - prox * probeta - proR * propsi) + 1 / 200))
        
              prolambda1 = gamma(len(datanorth) / 2 + 1, 1 / (propsi.T * iV * propsi + 1 / 200)) 
    
    
              prolambda0 = deepcopy(lambda0)
              proV = deepcopy(V)
              proiV = deepcopy(iV)
              prolambda0temp = uniform.rvs(0.01, 100)
              proVtemp = covkernel(datanorth, prolambda0temp)  
              proiVtemp = inv(proVtemp)    
              alpha = -0.5 * log(det(proVtemp)) - prolambda1 * propsi.T * proiVtemp * propsi + log(prolambda0 -0.01) \
                         + 0.5 * log(det(proV)) + prolambda1 * propsi.T * proiV * propsi - log(prolambda0temp - 0.01)   
              alpha = min(0 ,alpha)      
              u = uniform.rvs(0, 1) 
              if(alpha > log(u)):
                 prolambda0 = deepcopy(prolambda0temp)
                 proV = deepcopy(Vtemp)
                 proiV = deepcopy(iVtemp)
           
              kdie = mytree.death_number()
              
              lpropl = (n/2) * log(prolambda2) - prolambda2 * (proy - prox * probeta - proR * propsi).T * (proy - prox * probeta - proR * propsi) + \
                        (n/2) * log(prolambda1) - 0.5 * log(det(proV)) - prolambda1 * propsi.T * proiV * propsi + \
                         0.5 * log(det(prox.T * prox)) - probeta.T * (prox.T * prox) * probeta - prolambda2 / 200 - \
                          prolambda1 / 200 + log(prolambda0 - 0.01) + log(kdie)
              lcurpl = (n/2) * log(lambda2) - lambda2 * (y - x * beta - R * psi).T * (y - x * beta - R * psi) + \
                        (n/2) * log(lambda1) - 0.5 * log(det(V)) - lambda1 * psi.T * iV * psi + \
                         0.5 * log(det(x.T * x)) - beta.T * (x.T * x) * beta - lambda2 / 200 - \
                          lambda1 / 200 + log(lambda0 - 0.01) + log(k - 1)
           
             alpha = min(0, lpropl - lcurpl)
             u = uniform.rvs(0, 1)
             if(alpha > log(u)):
                 mytree = deepcopy(protree)
                 lambda0 = deepcopy(prolambda0)
                 lambda1 = deepcopy(prolambda1)
                 lambda2 = deepcopy(prolambda2)
                 V = deepcopy(proV)
                 iV = deepcopy(proiV)
                 beta = deepcopy(probeta)
                 psi = deepcopy(propsi)
                 R = deepcopy(proR)
                 x = deepcopy(prox)
                 y = deepcopy(proy)           
        elif(bp + dp < u <= bp + dp + cnu):
            l = list()
            nodeavailable = protree.root.perturblist(l)
            nodecnu = choice(nodeavailable)
            var = nodecnu.var
            [lmin, lmax] = nodecnu.perturbrange(var)
            nobs = len([each for each in datanorth[var] if each >= lmin and each <= lmax])
            
            varnew = choice([each for each in varlist if each != var])
            nodecnu.var = varnew
            [lminnew, lmaxnew] = nodecnu.perturbrange(varnew)
            nobsnew = len([each for each in datanorth[varnew] if each >= lminnew and each <= lmaxnew])
            nodecnu.val = choice([each for each in datanorth[varnew] if each >= lminnew and each <= lmaxnew])            
            if(protree.number_check(datanorth) == 0 or nobs == 0 or nobsnew == 0):
                  pass
            else:
                  pronodedata = protree.treedata(datanorth)
                  proindex = list()
                  for i in range(len(pronodedata)):
                     proindex = np.append(proindex, pronodedata[i].index.values)
             
                  proR = np.matrix(np.zeros(shape = (len(proindex), len(proindex))))
                  for i in range(len(proindex)):
                     proR[i, proindex[i]] = 1
             
                  proxlist = list()
                  for i in range(len(pronodedata)):
                     proxlist.append(np.matrix(\
                        indvar[indvar.index.isin(pronodedata[i].index.values)].values))
                  temp = list()
                  for i in range(len(pronodedata)):
                      temp = np.matrix(block_diag(temp,proxlist[i]))
                  prox = deepcopy(temp[1:92, :])
                  proy = list()
                  for i in range(len(pronodedata)):
                     proy = np.append(proy, pronodedata[i].Lichen_Cover_Logit.values)
                  proy = np.matrix(proy).T
                  
                  proqbeta = (lambda2 + 1) * (prox.T * prox)
                  promubeta = np.matrix(cg(proqbeta, \
                                 lambda2 * prox.T * (proy - proR * psi))[0]).T
                  probetatemp = np.matrix(multivariate_normal.rvs(np.zeros(len(promubeta)), \
                                         proqbeta)).T
                  probeta = promubeta + np.matrix(cg(proqbeta, probetatemp)[0]).T
         
         
                  proqpsi = lambda2 * (proR.T * proR) + lambda1 * iV
                  promupsi =  np.matrix(cg(proqpsi, \
                                lambda2 * proR.T * (proy - prox * probeta))[0]).T
                  propsitemp = np.matrix(multivariate_normal.rvs(np.zeros(len(promupsi)), proqpsi)).T
                  propsi = promupsi + np.matrix(cg(proqpsi, promupsi)[0]).T
         
                  prolambda2 = gamma(len(datanorth) / 2 + 1, 
                              1 / ((proy - prox * probeta - proR * propsi).T * (proy - prox * probeta - proR * propsi) + 1 / 200))
            
                  prolambda1 = gamma(len(datanorth) / 2 + 1, 1 / (propsi.T * iV * propsi + 1 / 200)) 
        
        
                  prolambda0 = deepcopy(lambda0)
                  proV = deepcopy(V)
                  proiV = deepcopy(iV)
                  prolambda0temp = uniform.rvs(0.01, 100)
                  proVtemp = covkernel(datanorth, prolambda0temp)  
                  proiVtemp = inv(proVtemp)    
                  alpha = -0.5 * log(det(proVtemp)) - prolambda1 * propsi.T * proiVtemp * propsi + log(prolambda0 -0.01) \
                             + 0.5 * log(det(proV)) + prolambda1 * propsi.T * proiV * propsi - log(prolambda0temp - 0.01)   
                  alpha = min(0 ,alpha)      
                  u = uniform.rvs(0, 1) 
                  if(alpha > log(u)):
                     prolambda0 = deepcopy(prolambda0temp)
                     proV = deepcopy(Vtemp)
                     proiV = deepcopy(iVtemp)
               
                  
                  lpropl = (n/2) * log(prolambda2) - prolambda2 * (proy - prox * probeta - proR * propsi).T * (proy - prox * probeta - proR * propsi) + \
                            (n/2) * log(prolambda1) - 0.5 * log(det(proV)) - prolambda1 * propsi.T * proiV * propsi + \
                             0.5 * log(det(prox.T * prox)) - probeta.T * (prox.T * prox) * probeta - prolambda2 / 200 - \
                              prolambda1 / 200 + log(prolambda0 - 0.01) + log(nobs)
                  lcurpl = (n/2) * log(lambda2) - lambda2 * (y - x * beta - R * psi).T * (y - x * beta - R * psi) + \
                            (n/2) * log(lambda1) - 0.5 * log(det(V)) - lambda1 * psi.T * iV * psi + \
                             0.5 * log(det(x.T * x)) - beta.T * (x.T * x) * beta - lambda2 / 200 - \
                              lambda1 / 200 + log(lambda0 - 0.01) + log(nobsnew)
               
                  alpha = min(0, lpropl - lcurpl)
                  u = uniform.rvs(0, 1)
                  if(alpha > log(u)):
                     mytree = deepcopy(protree)
                     lambda0 = deepcopy(prolambda0)
                     lambda1 = deepcopy(prolambda1)
                     lambda2 = deepcopy(prolambda2)
                     V = deepcopy(proV)
                     iV = deepcopy(proiV)
                     beta = deepcopy(probeta)
                     psi = deepcopy(propsi)
                     R = deepcopy(proR)
                     x = deepcopy(prox)
                     y = deepcopy(proy)          
        else:
            l = list()
            nodeavailable = protree.root.perturblist(l)
            nodecnu = choice(nodeavailable)
            var = nodecnu.var
            [lmin, lmax] = nodecnu.perturbrange(var)
            nodecnu.val = choice([each for each in datanorth[var] if each >= lmin and each <= lmax])
            
            if(protree.number_check(datanorth) == 0):
                  pass 
            else:
                  pronodedata = protree.treedata(datanorth)
                  proindex = list()
                  for i in range(len(pronodedata)):
                     proindex = np.append(proindex, pronodedata[i].index.values)
             
                  proR = np.matrix(np.zeros(shape = (len(proindex), len(proindex))))
                  for i in range(len(proindex)):
                     proR[i, proindex[i]] = 1
             
                  proxlist = list()
                  for i in range(len(pronodedata)):
                     proxlist.append(np.matrix(\
                        indvar[indvar.index.isin(pronodedata[i].index.values)].values))
                  temp = list()
                  for i in range(len(pronodedata)):
                      temp = np.matrix(block_diag(temp,proxlist[i]))
                  prox = deepcopy(temp[1:92, :])
                  
                  proy = list()
                  for i in range(len(pronodedata)):
                     proy = np.append(proy, pronodedata[i].Lichen_Cover_Logit.values)
                  proy = np.matrix(proy).T
                  
                  proqbeta = (lambda2 + 1) * (prox.T * prox)
                  promubeta = np.matrix(cg(proqbeta, \
                                 lambda2 * prox.T * (proy - proR * psi))[0]).T
                  probetatemp = np.matrix(multivariate_normal.rvs(np.zeros(len(promubeta)), \
                                         proqbeta)).T
                  probeta = promubeta + np.matrix(cg(proqbeta, probetatemp)[0]).T
         
         
                  proqpsi = lambda2 * (proR.T * proR) + lambda1 * iV
                  promupsi =  np.matrix(cg(proqpsi, \
                                lambda2 * proR.T * (proy - prox * probeta))[0]).T
                  propsitemp = np.matrix(multivariate_normal.rvs(np.zeros(len(promupsi)), proqpsi)).T
                  propsi = promupsi + np.matrix(cg(proqpsi, promupsi)[0]).T
         
                  prolambda2 = gamma(len(datanorth) / 2 + 1, 
                              1 / ((proy - prox * probeta - proR * propsi).T * (proy - prox * probeta - proR * propsi) + 1 / 200))
            
                  prolambda1 = gamma(len(datanorth) / 2 + 1, 1 / (propsi.T * iV * propsi + 1 / 200)) 
        
        
                  prolambda0 = deepcopy(lambda0)
                  proV = deepcopy(V)
                  proiV = deepcopy(iV)
                  prolambda0temp = uniform.rvs(0.01, 100)
                  proVtemp = covkernel(datanorth, prolambda0temp)  
                  proiVtemp = inv(proVtemp)    
                  alpha = -0.5 * log(det(proVtemp)) - prolambda1 * propsi.T * proiVtemp * propsi + log(prolambda0 -0.01) \
                             + 0.5 * log(det(proV)) + prolambda1 * propsi.T * proiV * propsi - log(prolambda0temp - 0.01)   
                  alpha = min(0 ,alpha)      
                  u = uniform.rvs(0, 1) 
                  if(alpha > log(u)):
                     prolambda0 = deepcopy(prolambda0temp)
                     proV = deepcopy(Vtemp)
                     proiV = deepcopy(iVtemp)
               
                  
                  lpropl = (n/2) * log(prolambda2) - prolambda2 * (proy - prox * probeta - proR * propsi).T * (proy - prox * probeta - proR * propsi) + \
                            (n/2) * log(prolambda1) - 0.5 * log(det(proV)) - prolambda1 * propsi.T * proiV * propsi + \
                             0.5 * log(det(prox.T * prox)) - probeta.T * (prox.T * prox) * probeta - prolambda2 / 200 - \
                              prolambda1 / 200 + log(prolambda0 - 0.01)
                  lcurpl = (n/2) * log(lambda2) - lambda2 * (y - x * beta - R * psi).T * (y - x * beta - R * psi) + \
                            (n/2) * log(lambda1) - 0.5 * log(det(V)) - lambda1 * psi.T * iV * psi + \
                             0.5 * log(det(x.T * x)) - beta.T * (x.T * x) * beta - lambda2 / 200 - \
                              lambda1 / 200 + log(lambda0 - 0.01)
               
                  alpha = min(0, lpropl - lcurpl)
                  u = uniform.rvs(0, 1)
                  if(alpha > log(u)):
                     mytree = deepcopy(protree)
                     lambda0 = deepcopy(prolambda0)
                     lambda1 = deepcopy(prolambda1)
                     lambda2 = deepcopy(prolambda2)
                     V = deepcopy(proV)
                     iV = deepcopy(proiV)
                     beta = deepcopy(probeta)
                     psi = deepcopy(propsi)
                     R = deepcopy(proR)
                     x = deepcopy(prox)
                     y = deepcopy(proy)      
    treelist.append(mytree)
    treesizelist.append(mytree.size)
    lambda0list.append(lambda0)
    lambda1list.append(lambda1)
    lambda2list.append(lambda2)
    vpsdlist.append(np.all(np.linalg.eigvals(V) >= 0))

tsizefile = open('treesize.txt', 'w')
for i in range(len(treesizelist)):
    tsizefile.write('%d\n' %treesizelist[i])
tsizefile.close()

l0file = open('lambda0.txt','w')
for i in range(len(lambda0list)):
    l0file.write('%6.8f\n' % lambda0list[i])
l0file.close()

l1file = open('lambda1.txt','w')
for i in range(len(lambda1list)):
    l1file.write('%6.8f\n' % lambda1list[i])
l1file.close()

l2file = open('lambda2.txt','w')
for i in range(len(lambda2list)):
    l2file.write('%6.8f\n' % lambda2list[i])
l2file.close()

vpsdfile = open('vpsd.txt','w')
for i in range(len(vpsdlist)):
    vpsdfile.write('%d\n' % vpsdlist[i])
vpsdfile.close()






