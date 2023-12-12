import numpy as np
from scipy import integrate

def rk4(f,a,b,x0,N,kwargs):
    h = (b-a)/(N+1)
    t = np.arange(a,b+h,h)
    x = [x0]
    for i in range(1,len(t)):
        k1 = h*f(t[i-1],x[i-1],**kwargs)
        k2 = h*f(t[i-1]+h/2,x[i-1]+k1/2,**kwargs)
        k3 = h*f(t[i-1]+h/2,x[i-1]+k1/2,**kwargs)
        k4 = h*f(t[i-1]+h,x[i-1]+k1,**kwargs)
        x.append(x[i-1]+(k1+2*k2+2*k3+k4)/6)
    return t,np.array(x)

def tol_rk4(f,a,b,x0,N_max,tol,kwargs):
    N = 4
    while True:
        t,I0 = rk4(f,a,b,x0,N,kwargs)
        N *= 2
        t,I1 = rk4(f,a,b,x0,N,kwargs)
        
        if (np.abs(I1[-1]-I0[-1]) < np.ones_like(I1)*tol).all():
            break
        elif 2*N >= N_max:
            print('N max reached without reaching tolerence.')
            break
    return N,t,I1

def secant(func,x0,x1,epsilon):
    x0_list = []
    x1_list = []
    x2_list = []
    while abs((x0-x1)/x0) > epsilon:
        x2 = x1 - (x1-x0)*func(x1)/(func(x1)-func(x0))
        x0_list.append(x0)
        x1_list.append(x1)
        x2_list.append(x2)
        x0 = x1
        x1 = x2
    return x1

def integrate_data(x,fx,method='simpson_1by3'):
    w = np.ones_like(x)    
    if method == 'trapezoid':
        w[1:-1] = 2
    elif method == 'simpson_1by3':
        w[1:-1:2] = 4
        w[2:-1:2] = 2
    elif method == 'simpson_3by8':
        w[1:-1] *= 3
        w[3:-1:3] = 2
    else:
        print("Invalid method name. Use 'trapezoid','simpson_1by3','simpson_3by8'.")
    
    I = (x[-1]-x[0])*(w@fx)/np.sum(w)
    return I

def normalize(x,y):
    y_norm = []
    for i in range(np.shape(y)[1]):
        y_i = y[:,i]
        y_i_norm = y_i/np.sqrt(integrate.simpson(y_i**2,x))
        y_norm.append(y_i_norm)

    return np.array(y_norm).T