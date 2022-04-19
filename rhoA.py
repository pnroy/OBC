from ChoiceMC_OBC import ChoiceMC
import numpy as np
g=.1
rotor_chain = ChoiceMC(m_max=20, P=19,g=g, MC_steps=10,N=2,PBC=False)
rotor_chain.runNMM()
Ngrid=rotor_chain.Ngrid
#rho_beta_over2=rotor_chain.rho_beta_over2
rho_beta_over2=rotor_chain.rhobeta2_rhobeta2
rho_beta_over2=rotor_chain.rho_potential
delta_phi=rotor_chain.delta_phi
Z0=rotor_chain.Z0
#trace of B d
rhoA=np.zeros(Ngrid,float)
rhoB=np.zeros(Ngrid,float)
rho0=np.zeros((Ngrid,Ngrid),float)
normA=0.
normB=0.
psi0_square=0.

for i1 in range(Ngrid):
    rhoA[i1]=0.
    for i2 in range(Ngrid):
        psi0_square=0.
        for i1p in range(Ngrid):
            for i2p in range(Ngrid):
                psi0_square+=rho_beta_over2[i1p*Ngrid+i2p,i1*Ngrid+i2]
        psi0_square=psi0_square**2
        rho0[i1,i2]=psi0_square    
    rhoA[i1]+=(psi0_square)
    normA+=rhoA[i1]

for i2 in range(Ngrid):
    rhoB[i2]=0.
    for i1 in range(Ngrid):
        psi0_square=0.
        for i1p in range(Ngrid):
            for i2p in range(Ngrid):
                psi0_square+=rho_beta_over2[i1p*Ngrid+i2p,i1*Ngrid+i2]
        psi0_square=psi0_square**2    
    rhoB[i2]+=(psi0_square)
    normB+=rhoB[i2]

rho0A=np.zeros(Ngrid,float)
rho0A_norm=0.
for i1 in range(Ngrid):
    for i2 in range(Ngrid):
        rho0A[i1]+=rho0[i1,i2]
    rho0A_norm+=rho0A[i1]

print(rotor_chain.E0_nmm)
print(rotor_chain.e1_dot_e2)
rhoA_out=open('rhoa_g_'+str(g)+'.dat','w')
Vout=open('V'+str(g)+'.dat','w')
for i1 in range(Ngrid):
    for i2 in range(Ngrid):
        Vout.write(str(i1*delta_phi)+' '+str(i2*delta_phi)+' '+str(rotor_chain.potential[i1*Ngrid+i2])+' '+str(rotor_chain.rho_potential[i1*Ngrid+i2,i1*Ngrid+i2])+' '+str(rho0[i1,i2])+'\n')
#    rhoA_out.write(str(i1*delta_phi)+' '+str(rhoA[i1]/normA)+' '+str(rhoB[i1]/normB)+' '+str(rho0A[i1]/rho0A_norm)+'\n')
    rhoA_out.write(str(i1*delta_phi)+' '+str(rho0A[i1]/rho0A_norm/delta_phi)+'\n')
rhoA_out.close()
