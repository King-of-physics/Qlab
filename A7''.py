#!/usr/bin/env python
# coding: utf-8

# In[40]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from scipy.special import hermite
import math


# In[41]:


def Eig(a,b,n,v):
    x=np.linspace(a,b,n)
    h=(b-a)/(n+1)
    diag=[]; non_diag=[]
    for i in range(n):
        diag.append(1/(h**2)+v(x[i]))
    non_diag=[1/(-2*h**2)]*(n-1) #non-diag elements
    w, v = eigh_tridiagonal(diag, non_diag) #w=eigenvalue, v=eigenvector
    return w,v


# In[42]:


#a
n0=1000 #n is the number of internal grid points
a=-20;b=20
def pot(x):
    return (x**2)/2
w,v=Eig(a,b,n0,pot)
w_10=w[:10]

print(w_10) #first 10 eigenvalues


# In[43]:


def analy(x,n):
    r=hermite(n)
    o=len(x)
    Y=np.zeros(o)
    for i in range(o):
        Y[i]=(r(x[i])*np.exp(-x[i]**2/2)/(np.sqrt(2**n*math.factorial(n))))
    return x,Y


# In[44]:


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
def normalization(u): #this function normalizes the eigenfunctions
    U=u**2
    L=u/np.sqrt(simp(U,a,b))
    return L


# In[45]:


Fcn=[];Analy=[]
# plt.plot(x,V)
for i in range(5):
    tt0=np.linspace(a,b,n0)
    p=analy(tt0,i)
    plt.figure()
    Y=normalization(p[1])
    Analy.append(Y)
    OO=v[:,i] #choosing the i_th eigenvector
    Sol=normalization(OO)
    Fcn.append(Sol)
    tt0=np.linspace(a,b,n0)
    plt.plot(p[0],Y,label='n='+str(i+1)+' Anly')
    plt.plot(tt0,Sol,'--',label='n='+str(i+1)+' Num')
    plt.title('n='+str(i+1))
    plt.xlabel(r'$\xi$');plt.ylabel(r'u($ \xi $)')
    plt.xlim(-10,10)
    plt.grid();plt.legend()


# In[49]:


#prob dens
plt.figure()
print(len(Analy))
for i in range(5):
    plt.plot(tt0,abs(Fcn[i])**2,'+',label='n='+str(i+1))
    plt.plot(tt0,abs(Analy[i])**2,label='Analytical')
    plt.legend()
plt.grid()
plt.title('Probability Densities')
plt.xlim(-10,10)
plt.xlabel(r'$\xi$');plt.ylabel('$|u(ξ)|^2 $')
plt.show()


# In[ ]:





# In[47]:


#classical turning point
x0=np.sqrt(2*w[100])
P=lambda x: 1/(np.pi*np.sqrt(x0**2-x**2))


# In[48]:


plt.figure()
OO=v[:,100] 
Sol=normalization(OO)
Fcn.append(Sol)
tt0=np.linspace(a,b,n0)
plt.title('finding electron in ground state')
plt.plot(tt0,abs(Sol)**2,'',label=r'$n=u_{100}(\xi)$')
plt.plot(tt0,P(tt0),label='Classical prob')
plt.xlabel(r'$\xi$');plt.ylabel(r'P')
plt.legend()
plt.grid()
plt.show()


# In[33]:


#Analytical eigenvalues
E=lambda n:(n+1/2)*h_b*ome


# In[34]:


ome=5.5*10**14 #s-1
h_b=6.5821*10**(-16) #eV s
FDM=w_10*h_b*ome
n=np.linspace(0,4,5)
data={'n':n,'$E_{FDM}$':FDM[:5],'$E_{num}$':E(n)[:5]}
table=pd.DataFrame(data)
display(table)


# In[35]:


x_max=np.linspace(1,10,20)


# In[36]:


E0=[];E1=[]
for i in x_max:
    w,v=Eig(-i,i,500,pot)
    E0.append(w[0])
    E1.append(w[1])
plt.figure()
plt.plot(x_max,E0,label='Ground state')
plt.plot(x_max,E1,label='First Excited state')
plt.title('comparing ground and first excited state')
plt.grid();plt.legend()
plt.xticks(np.arange(0,10))
plt.xlabel('$x_{max}$');plt.ylabel('E')
plt.show()


# In[ ]:




