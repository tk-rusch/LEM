from scipy import integrate
import numpy as np

def FHN_rhs(t,x):
    I = 0.5
    a = 0.7
    b = 0.8
    eps = 1./50.
    dim1 = x[0] - (x[0]**3)/3. - x[1] + I
    dim2 = eps*(x[0] + a - b*x[1])

    out = np.stack((dim1,dim2)).T

    return out

def get_data(N,T=1000):
    data_x = []
    data_y = []
    for i in range(N):
        t = np.linspace(0,400,T+1)
        x0 = np.array([float(np.random.rand(1))*2.-1.,0.])
        sol = integrate.solve_ivp(FHN_rhs, [0,400], x0, t_eval=t)
        data_x.append(sol.y[0,:-1])
        data_y.append(sol.y[0,1:])

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x.reshape(N,T,1), data_y.reshape(N,T,1)
