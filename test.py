from ChoiceMC_OBC import ChoiceMC
import numpy as np
from sys import argv
import matplotlib as plt
#listg=[0.,.1,.25,.5,.75,1.,1.25,1.5,1.75,2.,3.,4.]
listg=[.75,1.,1.25]
MC_steps=1000
P=191
Eout=open('E0_vs_g_MC'+str(MC_steps)+'P'+str(P)+'.dat','w')
Corr_out=open('eiej_vs_g_MC'+str(MC_steps)+'P'+str(P)+'.dat','w')
listE0=[]
listeiej=[]
for g in listg:
    rotor_chain = ChoiceMC(m_max=5, P=P,g=g, T=.01,MC_steps=MC_steps,Nskip=1,Nequilibrate=0,N=10,PBC=False,PIGS=True)
    rotor_chain.runMC(orientationalCorrelations = True)
    Eout.write(str(g)+' '+str(rotor_chain.E_MC)+' '+str(rotor_chain.E_stdError_MC)+'\n')
    Corr_out.write(str(g)+' '+str(rotor_chain.eiej_MC)+' '+str(rotor_chain.eiej_stdError_MC)+'\n')
Eout.close()
Corr_out.close()
