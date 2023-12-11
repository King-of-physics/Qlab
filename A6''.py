#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd


# In[17]:


#part(a)
a=0;b=3
def f(x):
    return (1+x**2)
N=40
y0=1;y_d=0
def numerov(f,Y,a,b,N):
    h=(b-a)/(N)
    y1=h*y_d+y0
    Y[1]=y1
    def c(x):
        return 1-h**2/12*(f(x))
    x=np.linspace(a,b,N+1)
    C=[]
    for i in x:
        C.append(c(i))
    for i in range(1,N):
        y=((12-10*C[i])*Y[i]-C[i-1]*Y[i-1])/(C[i+1])
        Y.append(y)
    return x,Y
sol=num(f,[1,0],a,b,N)
plt.figure()
plt.plot(sol[0],sol[1])
plt.grid()


# In[18]:


def fn(x,u=np.array([0,0])):
    return np.array([u[1],(1+x**2)*u[0]])


# In[19]:


oo=solve_ivp(fn,[a,b],[1,0],rtol=0.000005)
print(len(oo.t))


# In[13]:


#part(b)
plt.figure()
plt.plot(oo.t,oo.y[0],c='black',label='Inbuilt')
plt.plot(sol[0],sol[1],'.',c='r',label='Numerov')
plt.xlabel('x');plt.ylabel('u(x)')
plt.grid();plt.legend()
plt.show()


# In[21]:


n1=17
num_2=numerov(f,[y0,0],a,b,n1)


# In[20]:


#part(c)
dat={r'$x_i$':num_2[0],'$u_{num}$':num_2[1],'$u_{inb}$':oo.y[0],'$\Delta u$':abs(num_2[1]-oo.y[0])}
table=pd.DataFrame(dat)
table


# In[16]:


#part(c)
k=np.arange(1,7,1)
N0=2**k
p=['o','--','.','*','+','*-']
plt.figure()
for i in range(len(N0)):
#     print(i)
    sol1=numerov(f,[y0,0],a,b,N0[i])
    plt.plot(sol1[0],sol1[1],p[i],label='for n= '+str(N0[i]))
plt.grid();plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:




