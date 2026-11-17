"""Open-source Python port of the original full fluid--Poisson Gaussian fission.

Two-thirds de-aliased pseudospectral Fourier discretization, conservative
(n,m=nu) evolution, fourth-order Runge--Kutta and damped Newton/GMRES solve
of phi_xx - P[exp(phi)] + n = 0 at every RK stage and final state.

This is the FIGURE 1 production solver, not an analytic KdV stand-in.
Reference production settings require substantial CPU time.
"""
from __future__ import annotations
from dataclasses import dataclass
import time,warnings
from inspect import signature
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres

# SciPy 1.11 and earlier use `tol`, whereas recent SciPy releases use
# `rtol`. Resolve the keyword ONCE instead of inspecting the signature at
# every Newton step (a full Figure 1 run uses many GMRES solves).
_GMRES_TOL_KEY = 'rtol' if 'rtol' in signature(gmres).parameters else 'tol'

@dataclass
class FissionConfig:
    N:int=8192
    L:float=2400.
    x0:float=-900.
    amplitude:float=.2
    width:float=45.
    cfl:float=.05
    safety:float=1.25
    final_time:float=1200.
    frames:int=600
    initial_momentum:str='rightgoing_spectral'
    newton_tol:float=1e-12
    newton_maxit:int=12
    gmres_tol:float=1e-11
    gmres_restart:int=40
    gmres_maxit:int=8
    min_line_step:float=1/128
    density_floor:float=1e-8
    sigma:float=.2
    p1:float=1.
    p2:float=.15
    p3:float=.02
    @property
    def lam(self):return np.sqrt(1+self.sigma*self.p1)

    @classmethod
    def quick(cls):
        """Small diagnostic run"""
        return cls(N=256,L=600,x0=-150,width=45,amplitude=.12,
                   cfl=.10,final_time=16,frames=45)

class Grid:
    def __init__(self,N,L):
        if N<8 or N%2:raise ValueError('N must be an even integer >= 8')
        self.N=N;self.L=L;self.dx=L/N
        self.x=(np.arange(N)-N//2)*self.dx
        modes=np.fft.fftfreq(N)*N
        self.k=2*np.pi*modes/L
        self.d1=1j*self.k.copy();self.d1[N//2]=0
        self.d2=-self.k**2
        cutoff=np.floor((N-1)/3)
        self.mask=(np.abs(modes)<=cutoff)

    def dealias(self,f):return np.fft.ifft(np.fft.fft(f)*self.mask).real
    def dx1(self,f):return np.fft.ifft(np.fft.fft(f)*self.d1).real
    def dx2(self,f):return np.fft.ifft(np.fft.fft(f)*self.d2).real


def pressure(n,c):
    y=n-1
    return c.p1*y+c.p2*y*y+c.p3*y**3

def pressure_prime(n,c):
    y=n-1
    return c.p1+2*c.p2*y+3*c.p3*y*y

def poisson_residual(phi,n,g):
    return g.dealias(g.dx2(phi)-g.dealias(np.exp(phi))+n)

def poisson_solve(n,phi_guess,g,c):
    phi=g.dealias(phi_guess)
    n=g.dealias(n)
    scale=max(1.,np.max(abs(n)))
    total_iterations=0
    for iteration in range(c.newton_maxit+1):
        res=poisson_residual(phi,n,g)
        rel=np.max(np.abs(res))/scale
        if rel<=c.newton_tol:
            return phi,dict(newton=iteration,gmres=total_iterations,residual=rel)
        if iteration==c.newton_maxit:break
        exp_phi=np.exp(phi)
        def action(v):
            low=g.dealias(v)
            high=v-low
            return g.dealias(g.dx2(low)-g.dealias(exp_phi*low))-high
        J=LinearOperator((g.N,g.N),matvec=action,dtype='float64')
        cbar=np.mean(exp_phi)
        def precondition(v):
            low=g.dealias(v)
            high=v-low
            return g.dealias(np.fft.ifft(np.fft.fft(low)/(g.d2-cbar)).real)-high
        P=LinearOperator((g.N,g.N),matvec=precondition,dtype='float64')
        count=[0]
        def callback(_r):count[0]+=1
        delta,info=gmres(J,-res,restart=min(g.N,c.gmres_restart),
                         maxiter=c.gmres_maxit,atol=0,
                         M=P,callback=callback,callback_type='pr_norm',
                         **{_GMRES_TOL_KEY: c.gmres_tol})
        total_iterations+=count[0]
        if not np.all(np.isfinite(delta)):raise RuntimeError('Nonfinite Poisson Newton correction')
        old_norm=np.max(abs(res))
        step=1.
        accepted=False
        while step>=c.min_line_step:
            trial=g.dealias(phi+step*delta)
            if np.max(abs(poisson_residual(trial,n,g)))<old_norm:
                phi=trial;accepted=True;break
            step/=2
        if not accepted:
            raise RuntimeError(f'Poisson Newton line search failed at residual {rel:.2e}, GMRES info={info}')
    raise RuntimeError('Poisson Newton tolerance not reached')


def rhs(n,m,guess,g,c):
    n=g.dealias(n);m=g.dealias(m)
    if np.min(n)<=c.density_floor:raise RuntimeError('Nonpositive density in rhs')
    phi,stats=poisson_solve(n,guess,g,c)
    flux=g.dealias(m*m/n+c.sigma*pressure(n,c))
    force=g.dealias(n*g.dx1(phi))
    return g.dealias(-g.dx1(m)),g.dealias(-g.dx1(flux)-force),phi,stats

def rk4(n,m,guess,dt,g,c):
    a,b,phi1,s1=rhs(n,m,guess,g,c)
    a2,b2,phi2,s2=rhs(g.dealias(n+dt*a/2),g.dealias(m+dt*b/2),phi1,g,c)
    a3,b3,phi3,s3=rhs(g.dealias(n+dt*a2/2),g.dealias(m+dt*b2/2),phi2,g,c)
    a4,b4,phi4,s4=rhs(g.dealias(n+dt*a3),g.dealias(m+dt*b3),phi3,g,c)
    new_n=g.dealias(n+dt*(a+2*a2+2*a3+a4)/6)
    new_m=g.dealias(m+dt*(b+2*b2+2*b3+b4)/6)
    if np.min(new_n)<=c.density_floor:raise RuntimeError('Density safety floor reached')
    phi5,s5=poisson_solve(new_n,phi4,g,c)
    stats=[s1,s2,s3,s4,s5]
    return new_n,new_m,phi5,dict(newton=max(s['newton'] for s in stats),
                            gmres=max(s['gmres'] for s in stats),
                            residual=max(s['residual'] for s in stats))

def solve(cfg:FissionConfig,progress=True):
    g=Grid(cfg.N,cfg.L)
    z=(g.x-cfg.x0+cfg.L/2)%cfg.L-cfg.L/2
    phi0=g.dealias(cfg.amplitude*np.exp(-(z/cfg.width)**2))
    n=g.dealias(g.dealias(np.exp(phi0))-g.dx2(phi0))
    if np.min(n)<=cfg.density_floor:raise RuntimeError('Invalid initial Gaussian density')
    if cfg.initial_momentum=='rightgoing_spectral':
        phase_speed=np.sqrt(cfg.sigma*cfg.p1+1/(1+g.k**2))
        m=g.dealias(np.fft.ifft(phase_speed*np.fft.fft(n-1)).real)
    elif cfg.initial_momentum=='rightgoing_longwave':
        m=g.dealias(cfg.lam*(n-1))
    elif cfg.initial_momentum=='at_rest':m=np.zeros_like(n)
    else:raise ValueError('Unsupported initial momentum')
    phi,s0=poisson_solve(n,phi0,g,cfg)
    char0=np.max(abs(m/n)+np.sqrt(1+cfg.sigma*pressure_prime(n,cfg)))
    dt_target=cfg.cfl*g.dx/(cfg.safety*char0)
    steps=max(1,int(np.ceil(cfg.final_time/dt_target)))
    dt=cfg.final_time/steps
    indices=np.unique(np.rint(np.linspace(0,steps,min(cfg.frames,steps+1))).astype(int))
    phis=np.empty((len(indices),cfg.N))
    ts=indices*dt
    mass0=g.dx*np.sum(n); momentum0=g.dx*np.sum(m)
    mass_drift=np.empty(len(indices));momentum_drift=np.empty(len(indices))
    poisson_err=np.empty(len(indices))
    current=0
    def record():
        nonlocal current
        phis[current]=phi
        mass_drift[current]=abs(g.dx*n.sum()-mass0)/max(abs(mass0),np.finfo(float).eps)
        momentum_drift[current]=abs(g.dx*m.sum()-momentum0)/max(abs(momentum0),np.finfo(float).eps)
        poisson_err[current]=np.max(abs(poisson_residual(phi,n,g)))/max(1.,np.max(abs(n)))
        current+=1
    record()
    start=time.monotonic()
    if progress:print(f'Gaussian fission solver: N={cfg.N}, dt={dt:.6g}, {steps} RK4 steps, {len(indices)} frames',flush=True)
    next_progress=max(1,steps//10)
    for j in range(1,steps+1):
        n,m,phi,st=rk4(n,m,phi,dt,g,cfg)
        if current<len(indices) and j==indices[current]:record()
        if progress and (j%next_progress==0 or j==steps):
            print(f'  {j}/{steps} ({100*j/steps:.0f}%), Poisson residual={st["residual"]:.2e}, time={time.monotonic()-start:.1f}s',flush=True)
    # MATLAB phiComoving(xi,t) = phi(xi+lambda*t,t).
    comoving=np.array([np.fft.ifft(np.fft.fft(p)*np.exp(1j*g.k*cfg.lam*t)).real
                       for p,t in zip(phis,ts)])
    return dict(x=g.x,time=ts,phi_comoving=comoving,dt=dt,lam=cfg.lam,
                mass_drift=mass_drift,momentum_drift=momentum_drift,
                poisson_residual=poisson_err,x0=cfg.x0,width=cfg.width,
                quick=(cfg.N==256))
