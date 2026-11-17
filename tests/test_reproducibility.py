"""Non-proprietary numerical regression checks for manuscript figures.

Run from repository root:
    python -m unittest discover -s tests -v
"""
import unittest
import numpy as np
from plasma_common import EOS, first_turning_point
from fig1 import omega_exact, omega_rk4
from fig2 import calculate as calc2
from fig3 import calculate as calc3
from fig4 import compute as calc4
from fig5 import calculate as calc5, Parameters
from fission_solver import FissionConfig, solve


class TestScientificRegressions(unittest.TestCase):
    def test_dispersion_linear_rk4(self):
        k=np.array([1,2,4,8,12])
        w=omega_exact(k,.2)
        w_num=omega_rk4(k,.2,.01)
        self.assertLess(np.max(abs(w_num-w)),2e-6)

    def test_fig2_exact_critical_branch(self):
        eos,d=calc2(41)
        i=np.argmin(abs(d['delta']))
        self.assertAlmostEqual(d['A_exact_compressive'][i],.0292901699788686,delta=2e-6)
        self.assertAlmostEqual(d['A_exact_rarefactive'][i],-.0306987432509982,delta=2e-6)
        self.assertAlmostEqual(eos.p2c,-6)

    def test_fig3_exact_reference(self):
        eos=EOS()
        plus,_=first_turning_point(eos,eos.lam+.0012,p3=35.21186973360886,
                                    polarity=1,r_min=.001,r_max=.75,n_grid=1400)
        minus,_=first_turning_point(eos,eos.lam+.0012,p3=35.21186973360886,
                                     polarity=-1,r_min=.001,r_max=.75,n_grid=1400)
        self.assertAlmostEqual(plus,.0292901699788686,delta=2e-6)
        self.assertAlmostEqual(minus,-.0306987432509982,delta=2e-6)

    def test_fig4_spatial_folds(self):
        *_,folds=calc4(301)
        self.assertAlmostEqual(folds[0][4],-.00294170654039139,delta=5e-8)
        self.assertAlmostEqual(folds[1][4],-.00280735630532058,delta=5e-8)

    def test_fig5_double_layer_and_plateau(self):
        d=calc5(Parameters())
        self.assertAlmostEqual(d['p_ref'],7.894395506263443,delta=1e-9)
        self.assertAlmostEqual(d['ell_ref'],12.15828418,delta=1e-5)
        slope=(d['widths'][-1]-d['widths'][-2])/(d['x_width'][-1]-d['x_width'][-2])
        self.assertAlmostEqual(slope,12.14958955,delta=2e-3)
        self.assertTrue(np.all(d['dlen']>0))
        self.assertTrue(np.all(np.isfinite(d['ell_dl'])))

    def test_fig1_quick_solver_constraints(self):
        c=FissionConfig.quick()
        c.final_time=2
        c.frames=8
        data=solve(c,progress=False)
        self.assertLess(max(data['poisson_residual']),1e-10)
        self.assertLess(max(data['mass_drift']),1e-10)
        self.assertEqual(data['phi_comoving'].shape,(8,c.N))


if __name__=='__main__':unittest.main()
