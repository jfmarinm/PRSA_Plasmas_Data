#!/usr/bin/env python3
"""Regenerate manuscript Figure 2: EOS map, exact Sagdeev and Gardner branches.

`python fig2.py` recomputes exact fluid--Poisson stationary branches.
Six full-PDE markers are HISTORICAL MEASUREMENTS embedded verbatim from the
MATLAB script, not calculated here. See README for provenance.
"""
from __future__ import annotations
import argparse,csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from plasma_common import EOS,first_turning_point
from figure_style import configure,style,letter,export,BLUE,COPPER,PURPLE,GREY

MEASURED_DELTA=np.array([-.5,0.,+.5])
MEASURED_PLUS=np.array([.0421565989172775,.0292901699788686,.0203156118655429])
MEASURED_MINUS=np.array([-.0210515948606153,-.0306987432509982,-.0448264236770363])


def calculate(n_delta=241):
    eos=EOS(p3=(1.2+2*np.sqrt(1.2)*8/3)/.2)
    lam=eos.lam
    V=.0012
    delta=np.linspace(-.9,.9,n_delta)
    alpha=(1+eos.sigma*(eos.p1+eos.p2c+delta))/lam
    Disc=alpha**2+6*8*V
    gard_plus=(-alpha+np.sqrt(Disc))/8
    gard_minus=(-alpha-np.sqrt(Disc))/8
    exact_plus=np.full_like(delta,np.nan)
    exact_minus=np.full_like(delta,np.nan)
    nt_plus=np.full_like(delta,np.nan)
    nt_minus=np.full_like(delta,np.nan)
    for j,d in enumerate(delta):
        exact_plus[j],nt_plus[j]=first_turning_point(eos,lam+V,p2=eos.p2c+d,polarity=+1)
        exact_minus[j],nt_minus[j]=first_turning_point(eos,lam+V,p2=eos.p2c+d,polarity=-1)
    return eos,dict(delta=delta,p2=eos.p2c+delta,alpha=alpha,
      A_exact_compressive=exact_plus,A_exact_rarefactive=exact_minus,
      A_gardner_compressive=gard_plus,A_gardner_rarefactive=gard_minus,
      n_turn_compressive=nt_plus,n_turn_rarefactive=nt_minus)


def plot(eos,T,out=None):
    configure()
    fig=plt.figure(figsize=(12.5,6))
    ax=fig.add_axes([.104,.17,.341,.745])
    p1=np.linspace(0,2.4,200)
    crit=-p1-1/eos.sigma
    ylo,yhi=-8.45,-3.25
    ax.fill_between(p1,ylo,crit,color=(.97,.92,.88),lw=0)
    ax.fill_between(p1,crit,yhi,color=(.87,.93,.96),lw=0)
    gard_tint=np.array((.92,.87,.96))
    for side in (-1,1):
        dest=np.array((.87,.93,.96) if side>0 else (.97,.92,.88))
        for k in range(24):
            t0,t1=k/24,(k+1)/24
            t=(t0+t1)/2
            blend=t*t*(3-2*t)
            ax.fill_between(p1,crit+side*.65*t0,crit+side*.65*t1,
                           color=(1-blend)*gard_tint+blend*dest,lw=0)
    ax.plot(p1,crit,color=PURPLE,lw=2.35)
    ax.axvline(eos.p1,color=(.2,.23,.27),lw=1.25,ls=':')
    ax.plot(np.ones_like(MEASURED_DELTA),eos.p2c+MEASURED_DELTA,
            'o',mfc=(.18,.2,.23),mec='white',ms=6.7)
    ax.plot([1],[eos.p2c],'o',mfc=PURPLE,mec='white',ms=7.5)
    ax.set(xlim=(0,2.4),ylim=(ylo,yhi),xlabel=r'$p_1$',ylabel=r'$p_2$')
    ax.set_xticks(np.arange(0,2.51,.5))
    ax.set_yticks(np.arange(-8,-3,1))
    style(ax,13)
    letter(ax,'a',(.03,.965),16)
    ax.text(.07,.86,r'KdV: $\alpha>0$',transform=ax.transAxes,color=BLUE,fontsize=13)
    ax.text(.07,.80,'compressive',transform=ax.transAxes,color=BLUE,fontsize=12)
    ax.text(.71,.13,r'KdV: $\alpha<0$',transform=ax.transAxes,color=COPPER,fontsize=13,ha='center')
    ax.text(.71,.07,'rarefactive',transform=ax.transAxes,color=COPPER,fontsize=12,ha='center')
    ax.text(.79,.39,r'mKdV: $\alpha=0$',transform=ax.transAxes,color=PURPLE,fontsize=11,ha='center')
    ax.text(.23,.45,'Gardner',transform=ax.transAxes,color=(.4,.27,.54),fontsize=11)
    ax.text(1.07,yhi-.13,r'$p_1=1$ slice',va='top',fontsize=11,color=(.2,.23,.27))
    ax=fig.add_axes([.540,.17,.423,.745])
    x=T['delta']
    ax.plot(x,T['A_exact_compressive'],lw=2.4,color=BLUE)
    ax.plot(x,T['A_exact_rarefactive'],lw=2.4,color=COPPER)
    ax.plot(x,T['A_gardner_compressive'],'--',lw=1.6,color=BLUE)
    ax.plot(x,T['A_gardner_rarefactive'],'--',lw=1.6,color=COPPER)
    ax.plot(MEASURED_DELTA,MEASURED_PLUS,'o',mfc=BLUE,mec='black',ms=7.5)
    ax.plot(MEASURED_DELTA,MEASURED_MINUS,'o',mfc=COPPER,mec='black',ms=7.5)
    ax.axvline(0,ls=':',lw=1,color=GREY)
    ax.axhline(0,ls=':',lw=.85,color=GREY)
    ax.set(xlim=(-.9,.9),ylim=(-.067,.067),xlabel=r'$\delta=p_2-p_{2,c}$',ylabel=r'$A$')
    ax.set_xticks(np.arange(-.9,.91,.3));ax.set_yticks(np.arange(-.06,.061,.02))
    style(ax,13);letter(ax,'b',(.03,.965),16)
    ax.text(.75,.77,'compressive',transform=ax.transAxes,color=BLUE,fontsize=12,ha='center')
    ax.text(.72,.12,'rarefactive',transform=ax.transAxes,color=COPPER,fontsize=12,ha='center')
    ax.text(.51,.94,r'$p_2=p_{2,c}$',transform=ax.transAxes,color=GREY,fontsize=11)
    ax.plot([],[],'k-',lw=2.2,label='exact Sagdeev')
    ax.plot([],[],'k--',lw=1.7,label='Gardner')
    ax.plot([],[],'ko',mfc='.4',ms=7.5,label='full fluid--Poisson')
    ax.legend(loc='lower left',fontsize=10.5)
    export(fig,'Fig_02',out)


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--output-dir',default=None)
    pa.add_argument('--n-delta',type=int,default=241)
    args=pa.parse_args()
    eos,T=calculate(args.n_delta)
    from figure_style import output_paths
    root=output_paths('Fig_02',args.output_dir)
    keys=list(T)
    with open(str(root)+'_branches.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(keys)
        w.writerows(zip(*(T[k] for k in keys)))
    with open(str(root)+'_markers.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['delta_p2','A_measured_compressive','A_measured_rarefactive'])
        w.writerows(zip(MEASURED_DELTA,MEASURED_PLUS,MEASURED_MINUS))
    ix=np.argmin(abs(T['delta']))
    print('Critical exact amplitudes',T['A_exact_compressive'][ix],T['A_exact_rarefactive'][ix])
    if np.any(~np.isfinite(T['A_exact_compressive'])) or np.any(~np.isfinite(T['A_exact_rarefactive'])):
        print('Warning: some scan points have no verified root within the search domain')
    plot(eos,T,args.output_dir)

if __name__=='__main__': main()
