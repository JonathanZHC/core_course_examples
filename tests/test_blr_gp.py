"""Numerical regression checks for Chapter 6.3; run with unittest discovery."""
import unittest
from pathlib import Path
import sys
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import casadi as ca
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal
from ex6_SysID.SysID_utils import Identifier_BLR, Identifier_GP, construct_gp_casadi_expression
from utils.env import Env, Dynamics
from ex6_SysID.gpmpc_utils import GPMPCController


def basis(k):
    return [lambda p: np.ones_like(p), lambda p: np.sin(k * p), lambda p: np.cos(k * p)]


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([-1., -.7, -.4, .3, .7, 1.])
        self.h = .01 * np.cos(3 * self.p) + np.array([.002, -.001, .003, 0, -.002, .001])
        self.q = np.linspace(-1.2, 1.2, 31)

    def test_blr_matches_function_space_conditioning_with_nonzero_vector_prior(self):
        phi = basis(3)
        mu0 = np.array([.01, -.02, .005])  # 1-D means used to broadcast incorrectly
        S0 = np.array([[.02, .001, 0], [.001, .01, .002], [0, .002, .03]])
        sn2 = .005**2
        model = Identifier_BLR(phi, sigma2=sn2, mu0=mu0, Sigma0=S0)
        model.fit(self.p, self.h)
        F = np.column_stack([f(self.p) for f in phi])
        T = np.column_stack([f(self.q) for f in phi])
        C = F @ S0 @ F.T + sn2 * np.eye(len(self.p))
        cross = T @ S0 @ F.T
        mean = T @ mu0 + cross @ np.linalg.solve(C, self.h - F @ mu0)
        cov = T @ S0 @ T.T - cross @ np.linalg.solve(C, cross.T)
        actual_mean, std = model.predict(self.q, include_noise=False)
        assert_allclose(actual_mean.ravel(), mean, atol=1e-12)
        assert_allclose(std.ravel()**2, np.diag(cov), atol=1e-12)
        assert_allclose(model.log_marginal_likelihood(self.p, self.h),
                        multivariate_normal.logpdf(self.h, mean=F @ mu0, cov=C), atol=1e-9)
        assert_allclose(model.predict(self.q)[1]**2 - std**2, sn2, atol=1e-15)

    def test_blr_gp_equivalence_under_identical_periodic_prior(self):
        sf, sn, k = .02, .005, 3
        blr = Identifier_BLR(basis(k), sigma2=sn**2, Sigma0=sf**2 / 2 * np.eye(3))
        class PeriodicGP(Identifier_GP):
            def _kernel(self, a, b):
                d = np.asarray(a).reshape(-1, 1) - np.asarray(b).reshape(1, -1)
                return self.signal_std**2 / 2 * (1 + np.cos(k * d))
        gp = PeriodicGP(signal_std=sf, noise_std=sn)
        gp.jitter = 0  # exact matching likelihood for this well-conditioned problem
        for m in (blr, gp): m.fit(self.p, self.h)
        for noise in (True, False):
            for a, b in zip(blr.predict(self.q, noise), gp.predict(self.q, noise)):
                assert_allclose(a, b, atol=1e-12)
        assert_allclose(blr.log_marginal_likelihood(self.p, self.h),
                        gp.log_marginal_likelihood(self.p, self.h), atol=1e-10)

    def test_gp_matches_dense_conditioning_and_evidence(self):
        gp = Identifier_GP(.3, .02, .005)
        gp.fit(self.p, self.h)
        K = .02**2 * np.exp(-.5 * ((self.p[:, None] - self.p) / .3)**2)
        C = K + (.005**2 + gp.jitter) * np.eye(len(self.p))
        cross = .02**2 * np.exp(-.5 * ((self.q[:, None] - self.p) / .3)**2)
        mean = cross @ np.linalg.solve(C, self.h)
        var = .02**2 - np.einsum('ij,ji->i', cross, np.linalg.solve(C, cross.T))
        m, s = gp.predict(self.q, False)
        assert_allclose(m.ravel(), mean, atol=1e-13)
        assert_allclose(s.ravel()**2, var, atol=1e-13)
        assert_allclose(gp.log_marginal_likelihood(self.p, self.h),
                        multivariate_normal.logpdf(self.h, cov=C), atol=1e-10)
        assert_allclose(gp.predict(self.q)[1]**2 - s**2, gp.noise_std**2, atol=1e-15)

    def test_casadi_and_numpy_match_and_slope_covariance_matches_finite_difference(self):
        gp = Identifier_GP(.3, .02, .005)
        gp.fit(self.p, self.h)
        C = gp._kernel(self.p, self.p) + (gp.noise_std**2 + gp.jitter) * np.eye(len(self.p))
        def posterior_cov(a, b):
            return (gp._kernel([a], [b]) - gp._kernel([a], self.p) @
                    np.linalg.solve(C, gp._kernel(self.p, [b]))).item()
        for include_noise in (True, False):
            h, var, dvar = construct_gp_casadi_expression(gp, include_noise)
            m, s = gp.predict(self.q, include_noise)
            assert_allclose([float(h(q)) for q in self.q], m.ravel(), atol=1e-12)
            assert_allclose([float(var(q)) for q in self.q], s.ravel()**2, atol=1e-12)
            eps = 1e-5
            for q in [-.7, 0., .6]:
                fd = (posterior_cov(q+eps,q+eps) - posterior_cov(q+eps,q-eps)
                      - posterior_cov(q-eps,q+eps) + posterior_cov(q-eps,q-eps)) / (4*eps**2)
                assert_allclose(float(dvar(q)), fd, rtol=2e-5, atol=1e-9)

    def test_gp_duplicates_nearly_noiseless_and_stale_hyperparameters(self):
        gp = Identifier_GP(.3, .02, 0)
        gp.fit(np.repeat(self.p, 2), np.repeat(self.h, 2))
        self.assertTrue(np.all(np.isfinite(gp.predict(self.q)[1])))
        gp.lengthscale = .4
        with self.assertRaisesRegex(ValueError, 'changed'): gp.predict(self.q)
        with self.assertRaisesRegex(ValueError, 'changed'): construct_gp_casadi_expression(gp)
        gp.fit(self.p, self.h)
        gp.optimize_hyperparameters(self.p, self.h, [.2, .4], [.01, .02])
        with self.assertRaisesRegex(ValueError, 'Fit'): gp.predict(self.q)
        gp.fit(self.p, self.h)
        self.assertTrue(np.all(np.isfinite(gp.predict(self.q)[0])))

    def test_prior_only_and_invalid_noise(self):
        for model in (Identifier_BLR(basis(3), sigma2=.01), Identifier_GP()):
            model.fit(np.array([]), np.array([]))
            assert_allclose(model.predict(self.q)[0], 0)
            self.assertTrue(np.all(model.predict(self.q)[1] > 0))
        with self.assertRaises(ValueError): Identifier_BLR(basis(3), sigma2=0).fit(self.p,self.h)
        with self.assertRaises(ValueError): Identifier_GP(lengthscale=0).fit(self.p,self.h)

    def test_posterior_coverage_under_repeated_prior_draws(self):
        # Independent prior-predictive draws: checks conditional variance calibration,
        # unlike checking coverage on just one deterministic terrain.
        rng = np.random.default_rng(42)
        for kind in ('blr', 'gp'):
            model = (Identifier_BLR(basis(3), sigma2=.005**2, Sigma0=.02**2/2*np.eye(3))
                     if kind == 'blr' else Identifier_GP(.3,.02,.005))
            q = np.array([0.])
            points = np.r_[self.p,q]
            if kind == 'blr':
                F = np.column_stack([f(points) for f in basis(3)])
                prior = F @ (.02**2/2*np.eye(3)) @ F.T
            else:
                prior = model._kernel(points,points)
            C = prior + .005**2*np.eye(len(points))
            draws = rng.multivariate_normal(np.zeros(len(points)),C,size=3000)
            cross = prior[-1,:-1]
            model.fit(self.p,draws[0,:-1])
            std = model.predict(q)[1].item()
            means = draws[:,:-1] @ np.linalg.solve(C[:-1,:-1],cross)
            z = (draws[:,-1]-means)/std
            self.assertLess(abs(np.mean(z*z)-1),.09)
            self.assertLess(abs(np.mean(np.abs(z)<2)-.9545),.02)


class MPCIntegrationTests(unittest.TestCase):
    def test_dynamics_variance_includes_shared_slope_cross_term(self):
        p = ca.MX.sym('p')
        slope, var = .4, 1e-8
        env = Env(3,np.array([0.,0.]),np.array([.5,0.]),
                  symbolic_h_mean_ext=ca.Function('h',[p],[slope*p]),
                  symbolic_h_cov_ext=ca.Function('hvar',[p],[0*p]),
                  symbolic_dh_cov_ext=ca.Function('dhvar',[p],[var+0*p]))
        dyn = Dynamics(env);dyn.build_stochastic_model(.1)
        rng = np.random.default_rng(1)
        slopes = rng.normal(slope,np.sqrt(var),300000)
        for u in [-8.,8.]:
            accel = u/np.sqrt(1+slopes**2)-9.81*slopes/(1+slopes**2)
            actual = float(dyn.dynamics_variance_function_cont([0,0],[u])[1,1])
            assert_allclose(actual,np.var(accel),rtol=.015)
            assert_allclose(float(dyn.dynamics_variance_function_disc([0,0],[u])[1,1]),.01*actual)

    def test_stage_zero_reference_and_solver_failure(self):
        ctrl = GPMPCController.__new__(GPMPCController)
        ctrl.N=2;ctrl.dim_states=2;ctrl.dim_inputs=1;ctrl.dt=.1;ctrl.beta=0
        ctrl.name='test';ctrl.n_saturated=0;ctrl.Sigma_x_log=[]
        ctrl.env=Mock(target_state=np.array([.5,0.]),state_lbs=np.array([-2.,-.6]),state_ubs=np.array([2.,.6]))
        ctrl.dynamics=Mock();ctrl.dynamics.one_step_forward.return_value=np.zeros(2)
        ctrl.propagate_uncertainty=Mock(return_value=[np.zeros((2,2))]*3)
        ctrl.solver=Mock();ctrl.solver.solve.return_value=2
        with self.assertRaisesRegex(RuntimeError,'status 2'): ctrl.compute_action(np.zeros(2),0)
        refs=[call.args[2] for call in ctrl.solver.set.call_args_list if call.args[:2]==(0,'yref')]
        assert_allclose(refs[0],[.5,0.,0.])


if __name__=='__main__': unittest.main()
