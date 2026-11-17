#!/usr/bin/env python3
"""Figure 4: reduced Gardner equilibria and phase portraits, exact fP markers.

Numerical translation of the author's MATLAB code. The plotted phase
orbits and separatrices belong to the REDUCED Gardner Hamiltonian, whereas
open circles in the top panels encode equilibria of the exact Bernoulli
closure. 
"""
from __future__ import annotations
import argparse,csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import minimize_scalar
from plasma_common import EOS,enthalpy,p_prime
from figure_style import configure,style,letter,export,output_paths,BLUE,COPPER,PURPLE,SLATE,GREY,LIGHT

ALPHAS=(-.35,0.,+.35)
US=(-.0036,-.0017,+.0012)
BETA=8.
GAMMA=1/(2*np.sqrt(1.2))
PHILIM=(-.105,.072)
YLIM=(-.012,.012)


def potential(phi,U,alpha,beta=BETA):
    return -.5*U*np.asarray(phi)**2+alpha*np.asarray(phi)**3/6+beta*np.asarray(phi)**4/12


def reduced_class(phi,U,alpha):
    jac=-alpha*np.asarray(phi)/2-2*BETA*np.asarray(phi)**2/3
    return np.where(jac < -1e-12,1,np.where(jac>1e-12,-1,0))


def exact_u(phi,p2,eos):
    """Solve Bernoulli(n=e^phi,M) = phi for M-lambda."""
    phi=np.asarray(phi,dtype=float)
    n=np.exp(phi)
    h=n-1
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        I=enthalpy(n,p2,eos.p3,eos.p1)
        # Original Matlab uses a 5th-order series near n=1 to avoid cancellation.
        small=np.abs(h)<.002
        series=(eos.p1*h+(2*p2-eos.p1)*h*h/2
                +(3*eos.p3-2*p2+eos.p1)*h**3/3
                +(-3*eos.p3+2*p2-eos.p1)*h**4/4
                +(3*eos.p3-2*p2+eos.p1)*h**5/5)
        I=np.where(small,series,I)
        m2=2*(phi+eos.sigma*I)/(-np.expm1(-2*phi))
        U=np.where((m2>0)&(abs(phi)>=1e-12),np.sqrt(np.maximum(m2,0))-eos.lam,np.nan)
    return U


def exact_class(phi,U,p2,eos):
    n=np.exp(phi)
    slope=p_prime(n,p2,eos.p3,eos.p1)
    Bp=(eos.lam+U)**2/n**3-eos.sigma*slope/n
    jac=n-1/Bp
    return np.where((Bp>0)&(jac < -1e-12),1,np.where((Bp>0)&(jac > 1e-12),-1,0))


def compute(n=1601):
    eos=EOS()
    phi=np.linspace(-.125,.125,n)
    p2s=eos.p2c+np.asarray(ALPHAS)*eos.lam/eos.sigma
    ureduced=np.stack([a*phi/2+BETA*phi**2/3 for a in ALPHAS],axis=1)
    uexact=np.stack([exact_u(phi,p2,eos) for p2 in p2s],axis=1)
    cl_reduced=np.stack([reduced_class(phi,ureduced[:,i],a) for i,a in enumerate(ALPHAS)],axis=1)
    cl_exact=np.stack([exact_class(phi,uexact[:,i],p2,eos) for i,p2 in enumerate(p2s)],axis=1)
    folds=[]
    for a,p2 in ((ALPHAS[0],p2s[0]),(ALPHAS[2],p2s[2])):
        phir=-3*a/(4*BETA)
        ur=-3*a*a/(16*BETA)
        minimum=minimize_scalar(lambda x: float(exact_u(x,p2,eos)),
                               bounds=(phir-.02,phir+.02),method='bounded',
                               options={'xatol':1e-12})
        folds.append((a,phir,ur,minimum.x,minimum.fun))
    return eos,phi,p2s,ureduced,uexact,cl_reduced,cl_exact,folds


def energy_level(ax,E,U,alpha,lim=PHILIM,color=(.73,.73,.73),lw=.7,N=650):
    coeff=[BETA/12,alpha/6,-U/2,0,-E]
    roots=np.roots(coeff)
    roots=np.sort(roots.real[np.abs(roots.imag)<1e-8])
    points=np.unique(np.r_[lim[0],roots[(roots>lim[0])&(roots<lim[1])],lim[1]])
    for lo,hi in zip(points[:-1],points[1:]):
        if hi-lo <1e-12 or E-potential((lo+hi)/2,U,alpha)<=0: continue
        x=np.linspace(lo,hi,N)
        y=np.sqrt(np.maximum(0,2*(E-potential(x,U,alpha))/GAMMA))
        ax.plot(x,y,color=color,lw=lw,zorder=1)
        ax.plot(x,-y,color=color,lw=lw,zorder=1)


def flow_arrow(ax,phi,y,U,alpha,color,lengthpx=7,widthpx=4):
    # Orient using true reduced spatial dynamics in pixel coordinates.
    vx=y
    vy=(U*phi-alpha*phi**2/2-BETA*phi**3/3)/GAMMA
    bounds=ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    wpx,hpx=bounds.width*ax.figure.dpi,bounds.height*ax.figure.dpi
    xscale=(ax.get_xlim()[1]-ax.get_xlim()[0])/max(wpx,1)
    yscale=(ax.get_ylim()[1]-ax.get_ylim()[0])/max(hpx,1)
    direction=np.array([vx/xscale,vy/yscale]); length=np.linalg.norm(direction)
    if length<1e-12 or not np.isfinite(length):return
    t=direction/length
    normal=np.array([-t[1],t[0]])
    point=np.array([phi,y])
    base=point-lengthpx*np.array([t[0]*xscale,t[1]*yscale])
    flank=.5*widthpx*np.array([normal[0]*xscale,normal[1]*yscale])
    poly=np.array([point,base+flank,base-flank])
    ax.add_patch(Polygon(poly,closed=True,facecolor=color,edgecolor='none',zorder=7))



def periodic_flow_arrow(ax,E,U,alpha,limits=PHILIM,y_limits=YLIM):
    """One subtle arrow on a regular reduced Hamiltonian energy curve."""
    coeff=[BETA/12,alpha/6,-U/2,0,-E]
    roots=np.roots(coeff)
    rr=np.sort(roots.real[np.abs(roots.imag)<1e-8])
    rr=rr[(rr>limits[0])&(rr<limits[1])]
    edges=np.unique(np.r_[limits[0],rr,limits[1]])
    candidates=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        if hi-lo<.009:continue
        mid=.56*lo+.44*hi
        available=E-potential(mid,U,alpha)
        if available<=0:continue
        yy=np.sqrt(2*available/GAMMA)
        if yy>.84*max(abs(np.asarray(y_limits))):continue
        candidates.append((hi-lo,mid,yy))
    if candidates:
        _,x,y=max(candidates)
        flow_arrow(ax,x,y,U,alpha,(.53,.54,.55),lengthpx=5.25,widthpx=3.075)

def separatrix(ax,saddle,U,alpha,lim=PHILIM,lw=2.1,N=1400,arrows=True,small_arrow=False):
    a=BETA/12; b=alpha/6; c=-U/2
    B=b/a+2*saddle
    C=c/a+2*(b/a)*saddle+3*saddle*saddle
    disc=B*B-4*C
    if disc<=0:return
    left,right=np.sort([(-B-np.sqrt(disc))/2,(-B+np.sqrt(disc))/2])
    if not left<saddle<right:return
    ranges=((left,saddle,COPPER),(saddle,right,PURPLE))
    large=max(saddle-left,right-saddle)
    for lo,hi,col in ranges:
        x=np.linspace(lo,hi,N)
        inside=np.maximum(0,(x-left)*(right-x))
        y=np.sqrt(2*a/GAMMA)*np.abs(x-saddle)*np.sqrt(inside)
        ax.plot(x,y,color=col,lw=lw,zorder=5)
        ax.plot(x,-y,color=col,lw=lw,zorder=5)
        if arrows and (hi-lo>=.48*large or small_arrow):
            xx=lo+.52*(hi-lo)
            yy=np.sqrt(2*a/GAMMA)*abs(xx-saddle)*np.sqrt(max(0,(xx-left)*(right-xx)))
            for sign in [1,-1]: flow_arrow(ax,xx,sign*yy,U,alpha,col)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',default=None)
    ap.add_argument('--n-equil',type=int,default=1601)
    ap.add_argument('--no-insets',action='store_true')
    a=ap.parse_args()
    eos,phi,p2s,ur,ue,cr,ce,folds=compute(a.n_equil)
    base=output_paths('Fig_04',a.output_dir)
    with open(str(base)+'_equilibria.csv','w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['alpha','p2','Phi_equilibrium','U_reduced','U_exact','spatial_class_reduced','spatial_class_exact'])
        for j,alpha in enumerate(ALPHAS):
            for i,p in enumerate(phi):w.writerow([alpha,p2s[j],p,ur[i,j],ue[i,j],cr[i,j],ce[i,j]])
    configure()
    fig=plt.figure(figsize=(8.33,6.57),facecolor='white')
    up=[(.090,.555,.393,.355),(.556,.555,.393,.355)]
    low=[(.079,.117,.265,.300),(.389,.117,.265,.300),(.699,.117,.265,.300)]
    colors=[BLUE,SLATE,COPPER]
    for k,pos in enumerate(up):
        ax=fig.add_axes(pos)
        active=[1] if k==0 else [0,2]
        for j in active:
            col=colors[j]
            for cl,ls in ((1,'-'),(-1,'--')):
                y=np.where(cr[:,j]==cl,phi,np.nan)
                ax.plot(ur[:,j],y,ls,color=col,lw=1.8)
            valid=np.where(np.isfinite(ue[:,j]) & (ce[:,j]!=0) & (abs(phi)>.004)
                          & (ue[:,j]>=-.0044)&(ue[:,j]<=.0078))[0]
            if len(valid):
                indices=valid[np.unique(np.rint(np.linspace(0,len(valid)-1,min(18,len(valid)))).astype(int))]
                ax.plot(ue[indices,j],phi[indices],'o',mfc='white',mec=col,ms=4.2,mew=.85)
        ax.plot(np.linspace(-.0044,0,100),np.zeros(100),color=GREY,lw=2)
        ax.plot(np.linspace(0,.0078,100),np.zeros(100),'--',color=GREY,lw=1.8)
        ax.axvline(0,ls=':',color=LIGHT,lw=.85)
        if k==1:
            phif=np.linspace(-.047,.047,250)
            ax.plot(-BETA*phif**2/3,phif,':',color='.42',lw=1.3)
            for j,(alpha,phir,urr,phie,uee) in enumerate(folds):
                col=colors[[0,2][j]]
                ax.plot(uee,phie,'o',mfc='white',mec='black',ms=11,mew=1.45)
                ax.plot(urr,phir,'x',color=col,ms=10,mew=2.5)
            ax.text(.04,.8,r'$\alpha=-0.35$',transform=ax.transAxes,color=BLUE,fontsize=12)
            ax.text(.04,.12,r'$\alpha=+0.35$',transform=ax.transAxes,color=COPPER,fontsize=12)
        else:
            ax.text(.72,.88,r'$\alpha=0$',transform=ax.transAxes,color=SLATE,fontsize=12)
        style(ax,12);ax.set(xlim=(-.0044,.0078),ylim=(-.125,.125),xlabel=r'$U=M-\lambda$')
        ax.set_xticks([-.004,0,.004,.008]);ax.set_yticks([-.10,-.05,0,.05,.10])
        if k==0:ax.set_ylabel(r'$\Phi_e$',fontsize=16)
        else:ax.set_yticklabels([])
        letter(ax,'ab'[k])
        if k==0:
            ax.plot([],[],'-',color=SLATE,lw=1.8,label='spatial centre')
            ax.plot([],[],'--',color=SLATE,lw=1.8,label='spatial saddle')
            ax.plot([],[],'o',color=SLATE,mfc='white',ms=4.5,label='exact Sagdeev')
            ax.plot([],[],'x',color=SLATE,ms=10,mew=2.5,label='reduced fold')
            ax.legend(loc='lower left',fontsize=9.5)
    xx=np.linspace(PHILIM[0],PHILIM[1],1200)
    for j,pos in enumerate(low):
        U=US[j];alpha=.35
        ax=fig.add_axes(pos)
        ax.set(xlim=PHILIM,ylim=YLIM)
        wg=potential(xx,U,alpha)
        emin=wg.min(); emax=wg.max()+GAMMA*max(abs(np.array(YLIM)))**2/2
        levels=np.linspace(emin+1e-7,.60*emax,22)
        for E in levels:energy_level(ax,E,U,alpha)
        for ind in (int(round(22*.30))-1,int(round(22*.66))-1):
            periodic_flow_arrow(ax,levels[ind],U,alpha)
        equilibria=[0.]
        D=(alpha/2)**2+4*(BETA/3)*U
        if D>=0:
            equilibria.extend([(-alpha/2+np.sqrt(D))/(2*BETA/3),(-alpha/2-np.sqrt(D))/(2*BETA/3)])
        for eq in equilibria:
            jac=(U-alpha*eq-BETA*eq**2)/GAMMA
            if jac>1e-11:separatrix(ax,eq,U,alpha)
        for eq in equilibria:
            jac=(U-alpha*eq-BETA*eq**2)/GAMMA
            if jac>1e-11:ax.plot(eq,0,'d',ms=7.2,mfc='white',mec='.12',mew=1.5,zorder=10)
            elif jac< -1e-11:ax.plot(eq,0,'o',ms=6.2,mfc=BLUE,mec=BLUE,zorder=10)
            else:ax.plot(eq,0,'o',ms=5.4,mfc='white',mec='black',zorder=10)
        ax.axhline(0,color='.47',lw=.6,zorder=0)
        style(ax,11);ax.set(xlim=PHILIM,ylim=YLIM,xlabel=r'$\Phi$')
        ax.set_xticks([-.1,-.05,0,.05]);ax.set_yticks([-.01,0,.01])
        if j==0:ax.set_ylabel(r'$Y=\partial_z\Phi$',fontsize=14)
        else:ax.set_yticklabels([])
        letter(ax,'cde'[j]);ax.text(.93,1.10,rf'$U={U:+.4f}$',transform=ax.transAxes,ha='right',va='top',fontsize=10)
        if not a.no_insets and j>=1:
            # Same relative insets as the MATLAB figure, reusing analytic orbits.
            rel=(.640,.10,.430*.7,.470*.7)
            posi=(pos[0]+rel[0]*pos[2],pos[1]+rel[1]*pos[3],rel[2]*pos[2],rel[3]*pos[3])
            ins=fig.add_axes(posi,facecolor='white',zorder=25)
            plim=((- .014,.008) if j==1 else (-.012,.011))
            ylim=(-.00055,.00055) if j==1 else (-.00027,.00027)
            ins.set(xlim=plim,ylim=ylim)
            saddle=(-alpha/2+np.sqrt(D))/(2*BETA/3) if j==1 else 0
            centres=[eq for eq in equilibria if (U-alpha*eq-BETA*eq**2)/GAMMA < -1e-11 and eq>saddle]
            if centres:
                rightCentre=centres[0]
                Es=potential(saddle,U,alpha);Ec=potential(rightCentre,U,alpha)
                for fraction in (.30,.62,.85):energy_level(ins,Ec+fraction*(Es-Ec),U,alpha,plim,(.77,.77,.77),.65)
            separatrix(ins,saddle,U,alpha,lim=plim,lw=1.55,arrows=True,small_arrow=True)
            for eq in equilibria:
                jac=(U-alpha*eq-BETA*eq**2)/GAMMA
                if plim[0]<=eq<=plim[1] and jac< -1e-11:
                    ins.plot(eq,0,'o',mfc=BLUE,mec=BLUE,ms=4.4,zorder=10)
            ins.plot(saddle,0,'d',mfc='white',mec='black',ms=5.2,zorder=10)
            ins.axhline(0,color=GREY,lw=.4)
            style(ins,6)
            if j==1:
                ins.set_xticks([-.01,0]); ins.set_yticks([-.0004,0,.0004]);ins.set_yticklabels(['-4','0','4'])
            else:
                ins.set_xticks([-.008,0,.008]);ins.set_yticks([-.0002,0,.0002]);ins.set_yticklabels(['-2','0','2'])
            ins.text(.03,.95,r'$10^{-4}$',transform=ins.transAxes,va='top',fontsize=7,color=SLATE)
    export(fig,'Fig_04',a.output_dir)
    print('Exact-fold equilibria:',folds)

if __name__=='__main__': main()
