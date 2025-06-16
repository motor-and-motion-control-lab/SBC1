from calendar import c
import numpy as np
import matplotlib.pylab as plt
from webcolors import CSS21; plt.ion()

class ContactSchedule:
    def __init__(self) -> None:
        self.sch = []
        self.finaltime = 0
        self.hl = 0.13
        self.hw = 0.04
        self.ff = np.array([self.hl,0,0])
        self.fb = np.array([-self.hl,0,0])
        self.N = 200
        self.dt = 0.01
        self.xhist = []
        self.uhist = []
        self.cx = np.zeros((2,self.N))
        self.cy = np.zeros((2,self.N))
        self.px = np.zeros(self.N-1)
        self.py = np.zeros(self.N-1)
        self.pxobjs = 0.0*np.ones(self.N-1)
        self.pyobjs = 0.0*np.ones(self.N-1)
    def pushContactList(self,duration,clist,cenpos):
        self.sch.append([self.finaltime,clist,cenpos])
        self.finaltime += duration
    # def insertContactList(self,time,clist,clp):
    #     for i in range(len(self.sch)):
    #         if time > self.sch[i][0]:
    #             self.sch.insert(i+1,[time,clist,clp])
    def getCurrentsid(self,time):
        for i in range(len(self.sch)):
            if time < self.sch[i][0]:
                return i-1
        return len(self.sch)-1
                
    def printInfo(self):
        # print('time','\t','clist','\t','clp')
        cll = []
        for i in range(len(self.sch)):
            cll.append(len(str(self.sch[i][1])))
        # print(max(cll))
        for i in range(len(self.sch)):
            # print(self.sch[i][0],'\t',self.sch[i][1],'\t',self.sch[i][2])
            print('%.2f'%self.sch[i][0],'\t','%-*s'%(max(cll)+4,self.sch[i][1]),self.sch[i][2])
    def pushDS2SSOnR(self,rf,lf):
        ff = self.ff
        fb = self.fb
        self.pushContactList(0.3,['t1','t2','t3','t4','t5','t6','t7','t8'],(rf+lf)/2)
        self.pushContactList(0.3,['t1','t2','t3','t4','t5','t7'],(2*rf+lf+ff)/3)
        self.pushContactList(0.3,['t1','t2','t3','t4'],(rf))
    def pushDS2SSOnL(self,rf,lf):
        ff = self.ff
        fb = self.fb
        self.pushContactList(0.3,['t1','t2','t3','t4','t5','t6','t7','t8'],(rf+lf)/2)
        self.pushContactList(0.3,['t1','t3','t5','t6','t7','t8'],(rf+2*lf+ff)/3)
        self.pushContactList(0.3,['t5','t6','t7','t8'],(lf))
    def LIPLQR(self,uobjs,final_xobj,x0):
        N = self.N
        dt = self.dt
        P = np.zeros((2,2,N))
        p = np.zeros((2,N))
        K = np.zeros((1,2,N-1))
        d = np.zeros(N-1)
        h = np.ones(N-1)

        Q = np.diag([0.0,0.0])
        Qn = np.diag([10.0,0.0])
        # R = np.array([0.01])# 0.1结尾就不会上翘，N=200就能看出收敛
        R = 1 #0.01
        g = 9.8
        P[:,:,N-1] = Qn
        p[:,N-1] = Qn@(-final_xobj)

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
            # print(gu,Guu)
            d[k] = (gu[0]/Guu[0][0])#.T#invGuu@gu
            P[:,:,k] = Gxx + K[:,:,k].T@Guu@K[:,:,k] - Gxu@K[:,:,k] - K[:,:,k].T@(Gxu.T)
            p[:,k] = gx - (K[:,:,k].T*gu + K[:,:,k].T@Guu*d[k] - Gxu*d[k]).reshape(2)

        #Forward rollout starting at x0
        xhist = np.zeros((2,N))
        xhist[:,0] = x0
        uhist = np.zeros(N-1)
        for k in range(N-1):
            discrete_omega_square = g*dt/h[k]
            A = np.array([[1,dt],[discrete_omega_square,1]])
            B = np.array([[0],[-discrete_omega_square]])
            uhist[k] = (-K[:,:,k]@xhist[:,k] - d[k])[0]
            xhist[:,k+1] = A@xhist[:,k] + (B*uhist[k]).reshape(2)
        return xhist,uhist
    def getxyft(self,time):
        sid = self.getCurrentsid(time)
        x = self.sch[sid][2][0]
        y = self.sch[sid][2][1]
        return x,y
    def getTraj(self,time,cx0,cy0):
        N = self.N
        dt = self.dt
        # self.pxobjs = 0.0*np.ones(N-1)
        final_cxobj = np.array([0.06,0])
        # self.pyobjs = 0.0*np.ones(N-1)
        final_cyobj = np.array([0.15,0])
        for i in range(N-1):
            self.pxobjs[i],self.pyobjs[i] = self.getxyft(time+i*dt)
        self.cx,self.px = self.LIPLQR(self.pxobjs,final_cxobj,cx0)
        self.cy,self.py = self.LIPLQR(self.pyobjs,final_cyobj,cy0)
    def plotTraj(self):
        plt.subplot(3,1,1)
        plt.plot(self.cx[0,:],'gray',label='position of CoM')
        # plt.show()
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(self.px,'blue',label='point of force') #point of force
        plt.plot(self.pxobjs,'yellow',label='objects for point of force') #objects for point of force
        plt.legend(prop = {'size':6})
        # plt.subplot(3,1,3)
        # plt.plot(h,'red',label='height')
        # plt.legend()
        plt.savefig('images/mpp2tfcls1/comx_trajectories1_mpp2tfcls1.png')

        plt.figure(2)
        plt.subplot(3,1,1)
        plt.plot(self.cy[0,:],'gray',label='position of CoM')
        # plt.show()
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(self.py,'blue',label='point of force') #point of force
        plt.plot(self.pyobjs,'yellow',label='objects for point of force') #objects for point of force
        plt.legend(prop = {'size':6})
        plt.savefig('images/mpp2tfcls1/comy_trajectories1_mpp2tfcls1.png')

        plt.figure(3)
        plt.subplot(2,1,1)
        plt.plot(self.cx[0,:],self.cy[0,:],'gray',label='position of CoM')
        plt.plot(self.px,self.py,'blue',label='point of force') #point of force
        plt.plot(self.pxobjs,self.pyobjs,'yellow',label='objects for point of force') #objects for point of force
        plt.legend(prop = {'size':6})
        # plt.show()
        # plt.legend()
        # plt.subplot(2,1,2)
        plt.savefig('images/mpp2tfcls1/com_trajectories1_mpp2tfcls1.png')
    def pushDST(self,time,fpoint):
        self.pushContactList(time,['t1','t2','t3','t4','t5','t6','t7','t8'],fpoint)


if __name__ == '__main__':
    # [1,['t1','t2']]
    # 0.19,0.15,0; 0.095,-0.15,0
    # 0.49,0.15,0; 0.19,-0.15,0
    cs1 = ContactSchedule()
    ff = cs1.ff
    fb = cs1.fb
    rf1 = np.array([0.06,-0.15,0])
    lf1 = np.array([0.06,0.15,0])
    # print(rf1+lf1) # all need to be divided by 2
    # print(rf1+lf1+cs1.ff)
    # print(rf1)
    # print(rf1+cs1.ff)
    rf2 = np.array([0.46,-0.15,0])
    lf2 = np.array([0.46,0.15,0])
    # print(rf1+cs1.ff+lf2+cs1.fb)
    # print(lf2)
    # print(lf2+rf2)

    # two steps: start and stop
    # cs1.pushContactList(0.3,['t1','t2','t3','t4','t5','t6','t7','t8'],(rf1+lf1)/2)
    # cs1.pushContactList(0.3,['t1','t2','t3','t4','t5','t6'],(rf1+lf1+ff)/2)
    # cs1.pushContactList(0.3,['t1','t2','t3','t4'],(rf1))
    # cs1.pushContactList(0.1,['t1','t2'],(rf1+ff))
    # cs1.pushContactList(0.2,['t1','t2','t5','t6'],(rf1+ff+lf2+fb)/2)
    # cs1.pushContactList(0.3,['t5','t6','t7','t8'],(lf2))
    # cs1.pushContactList(0.3,['t1','t2','t3','t4','t5','t6','t7','t8'],(lf2+rf2)/2)
    # # print(cs1.sch)
    # cs1.printInfo()

    cs1.pushDST(0.4,np.array([0.0,0.0,0]))
    cs1.pushDS2SSOnL(rf1,lf1)
    cs1.printInfo()

    # print(cs1.getCurrentsid(0))
    cx0 = np.array([0.0,0.0])
    cy0 = np.array([0.0,0.0])
    cs1.getTraj(0,cx0,cy0)
    cs1.plotTraj()
    # for i in range(cs1.N-1):
    #     print(cs1.getxyft(i*0.01))
    # print(len(cs1.sch))
    # print(cs1.sch[0][0])
    # print(cs1.getCurrentsid(0.2))
