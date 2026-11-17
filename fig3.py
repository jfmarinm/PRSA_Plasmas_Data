#!/usr/bin/env python3
"""Regenerate manuscript Figure 3, including archival full-PDE measurements.

The 6 markers are empirical values copied from the author's MATLAB script.
They are not outputs of this stationary numerical scan. Run `python fig3.py`.
"""
from __future__ import annotations
import argparse,csv
import numpy as np
import matplotlib.pyplot as plt
from plasma_common import EOS,first_turning_point
from figure_style import configure,style,letter,export,output_paths,BLUE,COPPER,GREY

MEASURED_P3=np.array([16.,24.,35.2118697336089])
MEASURED_PLUS=np.array([.0489328161872397,.0370202181803042,.0292901699788686])
MEASURED_MINUS=np.array([-.0539116190685881,-.0394597787346034,-.0306987432509982])


def calculate(n_p3=401):
    eos=EOS()
    p3=np.linspace(0,40,n_p3)
    beta=3*(eos.sigma*p3-eos.lam**2)/(2*eos.lam)
    V=.0012
    Ap=np.full_like(p3,np.nan)
    Am=np.full_like(p3,np.nan)
    np_=np.full_like(p3,np.nan)
    nm_=np.full_like(p3,np.nan)
    for i,p in enumerate(p3):
        Ap[i],np_[i]=first_turning_point(eos,eos.lam+V,p3=p,polarity=1,
                                         r_min=.001,r_max=.75,n_grid=1400)
        Am[i],nm_[i]=first_turning_point(eos,eos.lam+V,p3=p,polarity=-1,
                                         r_min=.001,r_max=.75,n_grid=1400)
    mkp=np.where(beta>1e-12,np.sqrt(np.maximum(0,6*V/np.where(beta>0,beta,np.nan))),np.nan)
    mkm=-mkp
    return eos,dict(p3=p3,beta_c=beta,A_exact_compressive=Ap,A_exact_rarefactive=Am,
                    A_mkdv_compressive=mkp,A_mkdv_rarefactive=mkm,
                    n_turn_compressive=np_,n_turn_rarefactive=nm_)


def plot(eos,T,out):
    configure()
    fig=plt.figure(figsize=(11.4,5.4))
    ax=fig.add_axes([.092,.165,.555,.765])
    ax.plot(T['p3'],np.where(abs(T['A_exact_compressive'])<=.1,T['A_exact_compressive'],np.nan),'-',color=BLUE,lw=2.4)
    ax.plot(T['p3'],np.where(abs(T['A_exact_rarefactive'])<=.1,T['A_exact_rarefactive'],np.nan),'-',color=COPPER,lw=2.4)
    ax.plot(T['p3'],np.where(abs(T['A_mkdv_compressive'])<=.1,T['A_mkdv_compressive'],np.nan),'k--',lw=1.3)
    ax.plot(T['p3'],np.where(abs(T['A_mkdv_rarefactive'])<=.1,T['A_mkdv_rarefactive'],np.nan),'k--',lw=1.3)
    ax.plot(MEASURED_P3,MEASURED_PLUS,'o',mfc=BLUE,mec='black',ms=7)
    ax.plot(MEASURED_P3,MEASURED_MINUS,'o',mfc=COPPER,mec='black',ms=7)
    ax.axvline(eos.p3c,color=GREY,linestyle=':',lw=1)
    ax.axhline(0,color=GREY,linestyle=':',lw=.9)
    ax.set(xlim=(0,40),ylim=(-.1,.1),xlabel=r'$p_3$',ylabel=r'$A$')
    ax.set_xticks(np.arange(0,41,5));ax.set_yticks(np.arange(-.1,.101,.05))
    style(ax,15);letter(ax,'a',(.035,.95),18)
    ax.text(.67,.80,'compressive',color=BLUE,transform=ax.transAxes,fontsize=13,ha='center')
    ax.text(.69,.18,'rarefactive',color=COPPER,transform=ax.transAxes,fontsize=13,ha='center')
    ax.plot([],[],'k-',lw=2.2,label='exact Sagdeev')
    ax.plot([],[],'k--',lw=1.4,label='critical mKdV')
    ax.plot([],[],'ko',mfc='.4',ms=7,label='full fluid--Poisson')
    ax.legend(loc='lower left',fontsize=9.4)
    ax=fig.add_axes([.738,.165,.225,.765])
    ax.axvspan(0,eos.p3c,color=(.935,.961,.986),zorder=0)
    ax.axvspan(eos.p3c,40,color=(.995,.953,.911),zorder=0)
    ax.plot(T['p3'],T['beta_c'],color=BLUE,lw=2.3)
    ax.axvline(eos.p3c,color='.3',lw=1.4,ls=':')
    ax.axhline(0,color=GREY,lw=.95,ls=':')
    ax.set(xlim=(0,40),ylim=(-2,10),xlabel=r'$p_3$',ylabel=r'$\beta_c$')
    ax.set_xticks(np.arange(0,41,10));ax.set_yticks(np.arange(-2,11,2))
    style(ax,15);letter(ax,'b',(.30,.95),18)
    ax.text(3,4.5,'defocusing',rotation=90,ha='center',va='center',fontsize=11,color=(.18,.34,.48))
    ax.text(25,8,'focusing',ha='center',va='center',fontsize=12,color=(.51,.27,.17))
    ax.text(27,7,r'$\beta_c>0$',ha='center',fontsize=11,color=(.51,.27,.17))
    export(fig,'Fig_03',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default=None)
    p.add_argument('--n-p3',type=int,default=401)
    a=p.parse_args()
    eos,T=calculate(a.n_p3)
    root=output_paths('Fig_03',a.output_dir)
    with open(str(root)+'_branches.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(T.keys());w.writerows(zip(*T.values()))
    with open(str(root)+'_markers.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['p3','A_measured_compressive','A_measured_rarefactive'])
        w.writerows(zip(MEASURED_P3,MEASURED_PLUS,MEASURED_MINUS))
    plot(eos,T,a.output_dir)
    idx=np.argmin(abs(T['p3']-35.21186973360886))
    print('Nearest-grid Sagdeev amplitudes:',T['A_exact_compressive'][idx],T['A_exact_rarefactive'][idx])

if __name__=='__main__':main()
