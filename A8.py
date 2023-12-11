#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.special import hermite
import math


# In[2]:


x0=0;tol=0.5*10**(-5)
def v(x):
    return x**2/2


# In[3]:


def simp(array,a,b):
        n=len(array)
        simp_sum=array[0]+array[-1]
        for i in range (1,n):
            if i%2==0:
                simp_sum+=2*array[i]
            else:
    
                simp_sum+=4*array[i]
        simp_sum*=(b-a)/(3*(n))
        return simp_sum  
def normalization(u,a,b): #this function normalizes the eigenfunctions
    U=u**2
    L=u/np.sqrt(2*simp(U,a,b))
    return L


# In[4]:


def numerov(C,Y,a,b,N):
    h=(b-a)/(N)
    x=np.linspace(a,b,N+1)
    for i in range(1,N-1):
        y=((12-10*C[i])*Y[i]-(C[i-1]*Y[i-1]))/(C[i+1])
        Y.append(y)
    return x,Y
def analy(x,n): #Analytical solutions for Harmonic Oscillator
    r=hermite(n)
    o=len(x)
    Y=np.zeros(o)
    for i in range(o):
        Y[i]=(r(x[i])*np.exp(-x[i]**2/2)/(np.sqrt(2**n*math.factorial(n))))
    return x,Y   


# In[5]:


def shooting_num(v,x_max,N,max_nodes,n_iter,tol,prob):
    dx = x_max/N #step size
    ENERGY=[]; ANALY=[];Numerical=[]
    ddx12 = dx**2/12
    x = np.zeros(N+1); V = np.zeros(N+1)
    x[0] = x0; V[0] = v(x0)
    for i in range (1,N+1):
        x[i] = x0+i*dx
        V[i] = v(x[i])
    for i in range(max_nodes):
        n_nodes=i #for i number of maximym nodes required
        if n_nodes%2==0:
            h_nodes=n_nodes/2
            parity='even'
        else:
            h_nodes=(n_nodes)/2+0.5
            parity='odd'
            print(h_nodes,'h')
        if n_nodes > max_nodes:
            raise Exception('increase max_nodes')
        e_max = max(V) ; e_min = min(V)
        e = (e_min+e_max)/2 #first energy guess
        for k in range(0,N):
            i_cl=-1
            f = []; c = []
            for i in range(0,N):
                f_i = -2*(V[i]-e)
                f.append(f_i)
                c.append(1+ddx12*f_i)
                if abs(f_i) <= 10**(-20):
                    i_cl = i #classical turning point
            for i in range(1,len(f)):
                if np.sign(f[i])!= np.sign(f[i-1]):
                    i_cl=i
            if i_cl>=N-10:
                raise Exception('Change x_max')
            elif i_cl<1:
                raise Exception('No classical turning point found')
            if 2*h_nodes==n_nodes:
                u0=1   #initial conditions to be fed to numerov
                u1=(6-5*c[0])/c[1]
            else:
                u0=0; u1 = dx
            u=numerov(c,[u0,u1],0,x_max,N)[1]
            u=np.array(u)
            n_cross=0 #no. of nodes we have
            for i in range(1,N):
                if np.sign(u[i])!=np.sign(u[i-1]):
                    n_cross+=1
            if n_iter>1:
                if n_cross>h_nodes:
                    e_max=e
                else:
                    e_min=e
            e=0.5*(e_min+e_max)
            
            if abs(e_min-e_max)<=tol:
                break
            k_iter=k
        if k_iter==n_iter:
            print('Required tolerance could not be achieved for the maximum no. of iterations')
        else:
            print('Required tolerance achieved in '+str(k_iter)+' iterations')
    #NOrmalization
        P=normalization(u,0,x_max)
        print('Energy',e)
        P=u
        print('classical boundary ',x[i_cl])
        plt.figure()
        NUM=np.zeros(2*len(P)-1)
        for i in range(len(P),len(NUM)):
            NUM[i]=P[i-len(P)]
        X=np.linspace(-x_max,x_max,len(NUM))
        if parity=='even':
            for i in range(len(P)):
                NUM[i]=P[-i-1]
        elif parity=='odd':
            for i in range(len(P)):
                NUM[i]=-P[-i-1]            
        plt.plot(X,NUM,c='b',label='Numerov')
        plt.axhline(y=0)
        plt.axvline(x=x[i_cl],c='r',label='classical boundary') 
        Numerical.append(NUM)
        plt.axvline(x=0)
        state=round(e-1/2)
        plt.title('for n= '+str(state))
        X=np.linspace(-x_max,x_max,len(NUM))
        oo=analy(X,state)
        ANALY.append(oo[1])
        plt.xlabel('x');plt.ylabel(f'$u_{state}(x)$')
        plt.axvline(x=-x[i_cl],c='r')
        plt.plot(X,oo[1],c='cyan',label='analytical')
        plt.grid();plt.legend()
        plt.ylim(-1.5,1.5)
        ENERGY.append(e)
    if prob=='yes':
        plt.figure()
        for i in range(5):
            plt.plot(X,abs(ANALY[i])**2,label='Analytical')
            plt.plot(X,abs(Numerical[i])**2,'+',label='n='+str(i))
        plt.grid();plt.legend()
        plt.ylim(0,2)
        plt.xlabel('x');plt.ylabel('$|u(x)|^2$')
    return x,V,i_cl,ENERGY


# In[6]:


x,V,i_cl,E=(shooting_num(v,10,1000,5,2000,tol,'yes'))


# In[7]:


import pandas as pd
n= np.linspace(0,4,5)
AN=[]
for i in range(5):
    AN.append(i+1/2)
data={'n':n,'$E_{num}$': E,'$E_{Analytic}$': AN}
table=pd.DataFrame(data)
print(table)


# In[8]:


plt.figure()
plt.plot(n,E,'o')
plt.grid()
plt.xlabel('n');plt.ylabel('$e_n$')
def lsf(x,y): # least square fitting code
    if len(x) == len(y) :
        n = len(x)
        xy = [i*j for i,j in zip(x,y)]
        x_sq = [i**2 for i in x]
        sigma_x = sum(x)
        sigma_y = sum(y)
        sigma_xy = sum(xy)
        sigma_x_sq = sum(x_sq)
        slope = (n*sigma_xy - sigma_x*sigma_y)/(n*sigma_x_sq - sigma_x**2)
        intercept = (sigma_x_sq*sigma_y - sigma_x*sigma_xy)/(n*sigma_x_sq - sigma_x**2)
        return slope, intercept
    else:
        print('lists are not of same dimension.')
m,c=lsf(n,E)
plt.plot(n,m*n+c)
# plt.plot(n,AN)
print('Slope=     ',m)
print('Intercept= ',c)


# In[9]:


# for x_max=5
x1,V1,i_cl_1,E1=shooting_num(v,5,1000,5,2000,tol,'yes')

