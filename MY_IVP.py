import numpy as np
import matplotlib.pyplot as plt
def Euler_M(f,to,xo,tn,n):
    h=(tn-to)/(n)
    T=[to]
    mat=np.zeros((len(xo),n+1))
    mat[:,0]=xo
    for i in range(1,int(n+1)):
        x1=xo+h*f(to,xo)
        mat[:,i]=x1
        xo=x1
        to+=h
        T.append(to)
    return T,mat              

def Euler_tol(f,to,xo,tn,n_max,m,n_min):#linear polynomial n_min=3
    tol=0.5*10**(-m)
    k=0
    n=n_min
    mat= Euler_M(f,to,xo,tn,n)[1]
    M=mat[0][-1]
   
    while True:
        n=2*n
        mat_new=Euler_M(f,to,xo,tn,n)[1]
        M_new=mat_new[0][-1]
        E=M-M_new
        if M_new!=0.0001:
            if abs(E/M_new)<=tol or n>=n_max:
                break
        else:
            if abs(E)<=tol or n>=n_max:
                break
        M=M_new
        T,mat_new1=Euler_M(f,to,xo,tn,n)
    return n,T,mat_new1

def RK2(f,to,xo,tn,n):
    h=(tn-to)/(n)
    T=[to]
    mat=np.zeros((len(xo),n+1))
    mat[:,0]=xo
    for i in range(1,int(n+1)):
            K1=h*f(to,xo)
            K2=h*f(to+h,xo+K1)  
            x1=xo+(K1+K2)/2
            mat[:,i]=x1
            xo=x1
            to+=h
            T.append(to)
    return T,mat

def RK2_tol(f,to,xo,tn,n_max,m,n_min):#linear polynomial n_min=3
    tol=0.5*10**(-m)
    k=0
    n=n_min
    mat= RK2(f,to,xo,tn,n)[1]
    M=mat[0][-1]
   
    while True:
        n=2*n
        mat_new=RK2(f,to,xo,tn,n)[1]
        M_new=mat_new[0][-1]
        E=M-M_new
        if M_new!=0.0001:
            if abs(E/M_new)<=tol or n>=n_max:
                break
        else:
            if abs(E)<=tol or n>=n_max:
                break
        M=M_new
        T,mat_new1=RK2(f,to,xo,tn,n)
    return n,T,mat_new1
def RK_4(f,to,xo,tn,n):
    h=(tn-to)/(n)
    T=[to]
    mat=np.zeros((len(xo),n+1))
    mat[:,0]=xo
    for i in range(1,int(n+1)):
        k1=f(to,xo)
        k2=f(to+(h/2),xo+(h*k1/2))
        k3=f(to+(h/2),xo+(h*k2/2))
        k4=f(to+h,xo+h*k3)
        M_RK4=(k1+2*k2+2*k3+k4)/6
        x1=xo+h*M_RK4
        mat[:,i]=x1
        xo=x1
        to+=h
        T.append(to)
    return T,mat

def RK4_tol(f,to,xo,tn,n_max,m,n_min):#linear polynomial n_min=3
    tol=0.5*10**(-m)
    k=0
    n=n_min
    mat= RK4(f,to,xo,tn,n)[1]
    M=mat[0][-1]
   
    while True:
        n=2*n
        mat_new=RK4(f,to,xo,tn,n)[1]
        M_new=mat_new[0][-1]
        E=M-M_new
        if M_new!=0.0001:
            if abs(E/M_new)<=tol or n>=n_max:
                break
        else:
            if abs(E)<=tol or n>=n_max:
                break
        M=M_new
        mat_new1=RK4(f,to,xo,tn,n)
    return n,T,mat_new1
