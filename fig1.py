#!/usr/bin/env python3
"""Regenerate Figure 1: full fluid--Poisson Gaussian fission + dispersion.

Default `python fig1.py`: reproduce the MATLAB production parameters
(N=8192, 1200 time units, 600 frames). This can be computationally expensive.
`python fig1.py --quick`: short numerical smoke test.
`python fig1.py --input outputs/gaussian_fission.npz`: plot archived Python data.
Optional `--measurements-csv linear_dispersion_measurements.csv` with k,omega
uses measured values rather than the default independent linear-RK4 mode test.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from fission_solver import FissionConfig,solve
from figure_style import configure,style,letter,export,output_paths

KPOINTS=[1,2,4,8,12]
ORANGE=(.85,.325,.098)
BRIGHTBLUE=(.2021,.4788,.9911)

def omega_exact(k,ci2):
    k=np.asarray(k)
    return abs(k)*np.sqrt(ci2+1/(1+k*k))

def omega_rk4(k,ci2,dt):
    """Measure the discrete RK4 numerical phase of a right-going linear mode.

    MATLAB's unwrapped-phase least-squares slope is equivalent to the
    argument of its RK4 amplification polynomial for a pure eigenvector.
    """
    w=omega_exact(k,ci2)
    z=-1j*w*dt
    R=1+z+z*z/2+z**3/6+z**4/24
    return -np.angle(R)/dt

def plot(data,out_dir=None,measure_csv=None,quick=False):
    configure()
    lam=float(data['lam'])
    ci2=lam*lam-1
    t=np.asarray(data['time']);x=np.asarray(data['x']);z=np.asarray(data['phi_comoving'])
    # The original MATLAB crop depends on x0 and W.
    x0=float(data.get("x0", -150 if quick else -900))
    width=float(data.get("width",45))
    xi_lim=(50*round((x0-1.1*width)/50),50*round((x0+4.5*width)/50))
    if measure_csv:
        a=np.genfromtxt(measure_csv,delimiter=',',names=True)
        km=np.atleast_1d(a['k']);wm=np.atleast_1d(a['omega'])
        label='numerical data (CSV)'
    else:
        km=np.asarray(KPOINTS)
        wm=omega_rk4(km,ci2,float(data['dt']))
        label='linear RK4 frequency'
    fig=plt.figure(figsize=(11.2,4.55),facecolor='white')
    ax1=fig.add_axes([.075,.165,.335,.735])
    # imagesc(x,t,phiComoving') with time ascending
    image=ax1.imshow(z,origin='lower',aspect='auto',interpolation='nearest',
                     extent=(x[0],x[-1]+(x[1]-x[0]),t[0],t[-1]),
                     cmap='viridis',vmin=0,vmax=.35)
    ax1.set(xlim=xi_lim,ylim=(t[0],t[-1]),xlabel=r'$\bar\xi=x-\lambda t$',ylabel=r'$t$')
    ax1.set_xticks(np.arange(np.ceil(xi_lim[0]/50)*50,xi_lim[1]+1,50))
    ymax=200*np.floor(t[-1]/200)
    if ymax>0: ax1.set_yticks(np.arange(0,ymax+1,200))
    style(ax1,14);letter(ax1,'a',(.035,.955),18,color='white')
    cb=fig.colorbar(image,cax=fig.add_axes([.425,.165,.018,.735]))
    cb.set_ticks(np.arange(0,.351,.05));cb.ax.tick_params(labelsize=12)
    cb.ax.set_title(r'$\phi$',fontsize=16)
    ax2=fig.add_axes([.545,.165,.425,.735])
    kk=np.linspace(0,13,700)
    ax2.plot(kk,omega_exact(kk,ci2),color='black',lw=1.9,label='theory')
    ax2.plot(km,wm,'o',mfc='none',mec=ORANGE,ms=8.5,mew=1.5,label='numerics')
    style(ax2,14,grid=True)
    ax2.minorticks_on();ax2.grid(which='minor',alpha=.08)
    ax2.set(xlim=(0,14),ylim=(0,6),xlabel=r'$k$',ylabel=r'$\omega$')
    ax2.set_xticks(np.arange(0,15,2));ax2.set_yticks(np.arange(0,7,1))
    letter(ax2,'b',(.035,.965),18)
    ax2.legend(loc='lower right',fontsize=11.5,frameon=True)
    quick=bool(data.get("quick",quick))
    out_name='Fig_01_QUICK' if quick else 'Fig_01'
    export(fig,out_name,out_dir)
    with open(str(output_paths(out_name,out_dir))+'_dispersion_data.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['k','omega_numerical','omega_analytic','difference','provenance'])
        for k,wn in zip(km,wm):w.writerow([k,wn,float(omega_exact(k,ci2)),wn-float(omega_exact(k,ci2)),label])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='Run a reduced-resolution smoke test')
    parser.add_argument('--input',type=Path,default=None,help='Previously generated .npz simulation state')
    parser.add_argument('--measurements-csv',type=Path,default=None)
    parser.add_argument('--output-dir',type=Path,default=None)
    a=parser.parse_args()
    if a.input:
        with np.load(a.input) as f:data={k:f[k] for k in f.files}
    else:
        cfg=FissionConfig.quick() if a.quick else FissionConfig()
        data=solve(cfg)
        p=output_paths('gaussian_fission_QUICK' if a.quick else 'gaussian_fission',a.output_dir)
        np.savez_compressed(str(p)+'.npz',**data)
        print('Saved full numerical state:',str(p)+'.npz')
    plot(data,a.output_dir,a.measurements_csv,a.quick)

if __name__=='__main__':main()
