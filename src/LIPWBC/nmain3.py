# from cProfile import label
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

#Dynamic Programming Solution for P and K
N = 200
dt = 0.01
P = np.zeros((2,2,N))
p = np.zeros((2,N))
K = np.zeros((1,2,N-1))
d = np.zeros(N-1)
h = np.ones(N-1)
uobjs = 0.0*np.ones(N-1)
# for i in range(N-1):
#     h[i] += 0.01*i
# for i in range(50,100):
#     uobjs[i] = 0.01
uobjs[50 :100] = 0.1*np.ones(50)
uobjs[100:150] = -0.1*np.ones(50)
uobjs[150:199] = 0.0*np.ones(49)

Q = np.diag([0.0,0.0])
Qn = np.diag([10.0,0.0])
# R = np.array([0.01])# 0.1结尾就不会上翘，N=200就能看出收敛
R = 0.01
g = 9.8
final_xobj = np.array([0.0,0])
P[:,:,N-1] = Qn
p[:,N-1] = Qn@(-final_xobj)
x0 = np.array([0.0,0.0])


# A = B = g/h[N-2]
# K[N-2] = (R + B*P[N-1]*B)/(B*P[N-1]*A)
# P[N-2] = Q + K[N-2]*R*K[N-2] + (A-B*K[N-2])*P[N-1]*(A-B*K[N-2])
#Backward Riccati recursion
for k in range(N-2, -1, -1):
    discrete_omega_square = g*dt/h[k]
    A = np.array([[1,dt],[discrete_omega_square,1]])
    B = np.array([[0],[-discrete_omega_square]])
    gx = Q@(-final_xobj)+ A.T@p[:,k+1]
    gu = R*(-uobjs[k])  + B.T@p[:,k+1]
    Gxx = Q + A.T@P[:,:,k+1]@A
    Gxu = A.T@P[:,:,k+1]@B
    Guu = R + B.T@P[:,:,k+1]@B
    # invGuu = np.linalg.inv(Guu)
    K[:,:,k] = (Gxu/Guu).T#invGuu@Gxu #实际是Gux
    d[k] = (gu/Guu).T#invGuu@gu
    P[:,:,k] = Gxx + K[:,:,k].T@Guu@K[:,:,k] - Gxu@K[:,:,k] - K[:,:,k].T@(Gxu.T)
    p[:,k] = gx - (K[:,:,k].T*gu + K[:,:,k].T@Guu*d[k] - Gxu*d[k]).reshape(2)

    # K[k] = (B*P[k+1]*A)/(R + B*P[k+1]*B)
    # P[k] = Q + K[k]*R*K[k] + (A-B*K[k])*P[k+1]*(A-B*K[k])

#Forward rollout starting at x0
xhist = np.zeros((2,N))
xhist[:,0] = x0
uhist = np.zeros(N-1)
for k in range(N-1):
    discrete_omega_square = g*dt/h[k]
    A = np.array([[1,dt],[discrete_omega_square,1]])
    B = np.array([[0],[-discrete_omega_square]])
    uhist[k] = -K[:,:,k]@xhist[:,k] - d[k]
    xhist[:,k+1] = A@xhist[:,k] + (B*uhist[k]).reshape(2)

plt.subplot(3,1,1)
plt.plot(xhist[0,:],'gray',label='position of CoM')
# plt.show()
plt.legend()
plt.subplot(3,1,2)
plt.plot(uhist,'blue',label='point of force') #point of force
plt.plot(uobjs,'yellow',label='objects for point of force') #objects for point of force
plt.legend(prop = {'size':6})
plt.subplot(3,1,3)
plt.plot(h,'red',label='height')
plt.legend()
plt.savefig('images/nmain/comy_trajectories_nmain3.png')
print(xhist[:,-1])