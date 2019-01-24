import time
import random
import numpy as np
import math
import copy
a=np.matrix([[1.2,3.1,3.1]])


class SVM:
      def __init__(self,data,kernel,maxIter,C,epsilon):
            self.trainData=data
            self.C=C  
            self.kernel=kernel
            self.maxIter=maxIter
            self.epsilon=epsilon
            self.a=[0 for i in range(len(self.trainData))]
            self.w=[0 for i in range(len(self.trainData[0][0]))]
            self.eCache=[[0,0] for i in range(len(self.trainData))]
            self.b=0
            self.xL=[self.trainData[i][0] for i in range(len(self.trainData))]
            self.yL=[self.trainData[i][1] for i in range(len(self.trainData))]

      def train(self):
            self.__SMO()
            self.__update()

      def __kernel(self,A,B):
            res=0
            if self.kernel=='Line':
                  res=self.__Tdot(A,B)
            elif self.kernel[0]=='Gauss':
                  K=0
                  for m in range(len(A)):
                       K+=(A[m]-B[m])**2 
                  res=math.exp(-0.5*K/(self.kernel[1]**2))
            return res


      def __Tdot(self,A,B):
            res=0
            for k in range(len(A)):
                  res+=A[k]*B[k]
            return res


      def __SMO(self):
            support_Vector=[]
            self.a=[0 for i in range(len(self.trainData))]
            pre_a=copy.deepcopy(self.a)
            for it in range(self.maxIter):
                  flag=1
                  for i in range(len(self.xL)):
                        diff=0
                        self.__update()
                        Ei=self.__calE(self.xL[i],self.yL[i])
                        j,Ej=self.__chooseJ(i,Ei)
                        (L,H)=self.__calLH(pre_a,j,i)
                        kij=self.__kernel(self.xL[i],self.xL[i])+self.__kernel(self.xL[j],self.xL[j])-2*self.__kernel(self.xL[i],self.xL[j])
                        if(kij==0):
                              continue
                        self.a[j] = pre_a[j] + float(1.0*self.yL[j]*(Ei-Ej))/kij
                        self.a[j] = min(self.a[j], H)
                        self.a[j] = max(self.a[j], L)
                        #self.a[j] = min(self.a[j], H)
                        #print L,H
                        self.eCache[j]=[1,self.__calE(self.xL[j],self.yL[j])]
                        self.a[i] = pre_a[i]+self.yL[i]*self.yL[j]*(pre_a[j]-self.a[j])
                        self.eCache[i]=[1,self.__calE(self.xL[i],self.yL[i])]
                        diff=sum([abs(pre_a[m]-self.a[m]) for m in range(len(self.a))])
                        #print diff,pre_a,self.a
                        if diff < self.epsilon:
                              flag=0
                        pre_a=copy.deepcopy(self.a)
                  if flag==0:
                        print it,"break"
                        break

      def __chooseJ(self,i,Ei):
            self.eCache[i]=[1,Ei]
            chooseList=[]
            for p in range(len(self.eCache)):
                  if self.eCache[p][0]!=0 and p!=i:
                        chooseList.append(p)
            if len(chooseList)>1:
                  delta_E=0
                  maxE=0
                  j=0
                  Ej=0
                  for k in chooseList:
                        Ek=self.__calE(self.xL[k],self.yL[k])
                        delta_E=abs(Ek-Ei)
                        if delta_E>maxE:
                              maxE=delta_E
                              j=k
                              Ej=Ek
                  return j,Ej
            else:
                  j=self.__randJ(i)
                  Ej=self.__calE(self.xL[j],self.yL[j])
                  return j,Ej

      def __randJ(self,i):
            j=i
            while(j==i):
                  j=random.randint(0,len(self.xL)-1)
            return j

      def __calLH(self,pre_a,j,i):
            if(self.yL[j]!= self.yL[i]):
                  return (max(0,pre_a[j]-pre_a[i]),min(self.C,self.C-pre_a[i]+pre_a[j]))
            else:
                  return (max(0,-self.C+pre_a[i]+pre_a[j]),min(self.C,pre_a[i]+pre_a[j]))

      def __calE(self,x,y):
            #print x,y
            y_,q=self.predict(x)
            return y_-y

      def __calW(self):
            self.w=[0 for i in range(len(self.trainData[0][0]))]
            for i in range(len(self.trainData)):
                  for j in range(len(self.w)):
                        self.w[j]+=self.a[i]*self.yL[i]*self.xL[i][j]

      def __update(self):
            self.__calW()
            #print self.a
            maxf1=-99999
            min1=99999
            for k in range(len(self.trainData)):
                  y_v=self.__Tdot(self.w,self.xL[k])
                  #print y_v
                  if self.yL[k]==-1:
                        if y_v>maxf1:
                              maxf1=y_v
                  else:
                        if y_v<min1:
                              min1=y_v
            self.b=-0.5*(maxf1+min1)

      def predict(self,testData):
            pre_value=0
            for i in range(len(self.trainData)):
                  pre_value+=self.a[i]*self.yL[i]*self.__kernel(self.xL[i],testData)
            pre_value+=self.b
            if pre_value<0:
                  y=-1
            else:
                  y=1
            return y,abs(pre_value-0)

      def save(self):
            pass

def LoadSVM():
      pass