"""Redesigned GP-MPC figure for the tutorial paper (Chapter 6.3), with parameter-study presets.

Each case re-runs the Chapter 6.3 experiment (tuned GP on the data with a gap, GP-MPC at
beta = 0, 1, 2, 3 plus the true-model reference) for its own cost weights and covariance
propagation, caches the closed-loop results, and draws the two-row figure:

    Problem Setup (true-scale scene + zoomed terrain/GP strip)
    Plan at t = 0 in the state space  |  Performance vs. Safety

Usage, from the repository root with the course Python environment (acados required):

    python ex6_SysID/tutorial_gp_mpc_figure.py --case baseline          # one case
    python ex6_SysID/tutorial_gp_mpc_figure.py --all                    # every case
    python ex6_SysID/tutorial_gp_mpc_figure.py --all --rerun            # ignore cached data

Outputs go to ex6_SysID/figures/<output>.{png,pdf}; cached closed-loop data to
ex6_SysID/figures/gp_mpc_redesign_data/<case>.npz. acados code generation is written to
ex6_SysID/, next to the notebook's own codegen folders.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
FIG_DIR = os.path.join(HERE, "figures")
DATA_DIR = os.path.join(FIG_DIR, "gp_mpc_redesign_data")

# --------------------------------------------------------------------------------------
# Cases. Q and R are the MPC weights (Q_f = Q, as in the notebook); `ancillary=False`
# propagates the covariance open loop (K = 0); `overlay` draws another case's plan in gray.
# --------------------------------------------------------------------------------------
CASES = {
    "baseline":        dict(Q=(5.0, 5.0), R=0.1,  ancillary=True,  cost_ylim=None,      output="tutorial_gp_mpc_redesign"),
    "Q50":             dict(Q=(5.0, 0.0), R=0.1,  ancillary=True,  cost_ylim=(35, 105), output="tutorial_gp_mpc_redesign_Q50"),
    "Q50R0":           dict(Q=(5.0, 0.0), R=1e-6, ancillary=True,  cost_ylim=(30, 70),  output="tutorial_gp_mpc_redesign_Q50R0"),
    "Q50R001":         dict(Q=(5.0, 0.0), R=0.01, ancillary=True,  cost_ylim=(30, 70),  output="tutorial_gp_mpc_redesign_Q50R001"),
    "Q50R001_noK":     dict(Q=(5.0, 0.0), R=0.01, ancillary=False, cost_ylim=(30, 70),  output="tutorial_gp_mpc_redesign_Q50R001_noK"),
    "Q50R001_overlay": dict(Q=(5.0, 0.0), R=0.01, ancillary=True,  cost_ylim=(30, 70),  output="tutorial_gp_mpc_redesign_Q50R001_overlay",
                            data_from="Q50R001", overlay="Q50R001_noK"),
}

# Experiment constants, identical to the notebook's setup cell
CASE_REAL, TERRAIN_PARAM = 3, 0.01
X0, XT = np.array([-0.5, 0.0]), np.array([0.5, 0.0])
V_MAX = 0.6
STATE_LBS, STATE_UBS = np.array([-2.0, -V_MAX]), np.array([2.0, V_MAX])
INPUT_LBS, INPUT_UBS = -8.0, 8.0
N, FREQ, T_TERMINAL = 20, 10, 8
GAP, NUM_SAMPLES, SIGMA_MEAS = (-0.5, 0.5), 150, 0.005
BETAS = [0, 1, 2, 3]


# --------------------------------------------------------------------------------------
# Closed-loop experiment
# --------------------------------------------------------------------------------------
def simulate_case(name):
    """Run the experiment for one case and save its results; returns the saved dict."""
    import casadi as ca
    from utils.env import Env, Dynamics
    from utils.simulator import Simulator
    from ex6_SysID.SysID_utils import GenerateData, Identifier_GP, construct_gp_casadi_expression
    from ex6_SysID.gpmpc_utils import GPMPCController

    cfg = CASES[name]
    Q, R = np.diag(cfg["Q"]), np.array([[cfg["R"]]])
    os.chdir(HERE)  # acados codegen lands next to the notebook's codegen folders

    env_real = Env(CASE_REAL, X0, XT, param=TERRAIN_PARAM, state_lbs=STATE_LBS, state_ubs=STATE_UBS,
                   input_lbs=INPUT_LBS, input_ubs=INPUT_UBS)
    dynamics_real = Dynamics(env_real)

    np.random.seed(49)
    data_gen = GenerateData(p_range=[(-2.0, GAP[0]), (GAP[1], 2.0)], num_samples=NUM_SAMPLES, case=CASE_REAL, param=TERRAIN_PARAM)
    data_gen.set_noise(mean=0.0, std=SIGMA_MEAS)
    p_train, h_train = data_gen.generate_data()
    true_func = data_gen.get_symbolic_function()

    gp = Identifier_GP(noise_std=SIGMA_MEAS)
    (l_opt, sf_opt), _ = gp.optimize_hyperparameters(p_train, h_train)
    gp.fit(p_train, h_train)
    h_gp, s2h, s2dh = construct_gp_casadi_expression(gp, include_noise=False)

    p_grid = np.linspace(-1.0, 1.0, 801)
    h_true = np.array([float(true_func(p)) for p in p_grid])
    gp_mean, gp_std = (a.ravel() for a in gp.predict(p_grid, include_noise=False))

    class Probe(GPMPCController):
        """The chapter's controller, plus: full covariance of the first plan, and tolerance of
        acados stopping at its iteration cap (status 2), in which case the returned iterate is
        used, as the committed controller did before the status check was added."""

        def propagate_uncertainty(self, mu_x_seq, mu_u_seq):
            Sigma_seq = super().propagate_uncertainty(mu_x_seq, mu_u_seq)
            if not hasattr(self, "Sigma_full_k0"):
                self.Sigma_full_k0 = np.array(Sigma_seq)
            return Sigma_seq

        def compute_action(self, current_state, current_time):
            try:
                return super().compute_action(current_state, current_time)
            except RuntimeError as e:
                if "status 2" not in str(e):
                    raise
                self.n_maxiter = getattr(self, "n_maxiter", 0) + 1
                x_pred = np.array([self.solver.get(i, "x") for i in range(self.N + 1)])
                u_pred = np.array([self.solver.get(i, "u") for i in range(self.N)])
                self.last_u_pred, self.last_x_pred = u_pred.copy(), x_pred.copy()
                return self.solver.get(0, "u"), x_pred, u_pred

    def closed_loop_cost(states, inputs):
        err = states - XT
        return float(np.sum(np.einsum("ij,jk,ik->i", err, Q, err)) + R[0, 0] * np.sum(inputs ** 2))

    def run(env_l, beta, solver_name):
        ctrl = Probe(env_l, Dynamics(env_l), Q, R, Q, N, FREQ, beta=beta, name=solver_name, verbose=False)
        if not cfg["ancillary"]:
            ctrl.K_ancillary = np.zeros((1, 2))
        sim = Simulator(dynamics_real, ctrl, env_real, 1 / FREQ, T_TERMINAL)
        sim.run_simulation()
        states, inputs = np.array(sim.state_traj), np.array(sim.input_traj)
        return dict(states=states, inputs=inputs, sigma=np.array(ctrl.Sigma_x_log), pred=np.array(sim.state_pred_traj),
                    Sigma_full_k0=ctrl.Sigma_full_k0, cost=closed_loop_cost(states, inputs),
                    violation=max(0.0, np.abs(states[:, 1]).max() - V_MAX), nsat=ctrl.n_saturated,
                    nmaxiter=getattr(ctrl, "n_maxiter", 0))

    p_sym = ca.MX.sym("p")
    zero = ca.Function("zero", [p_sym], [ca.MX(0.0)])
    env_true = Env(CASE_REAL, X0, XT, symbolic_h_cov_ext=zero, symbolic_dh_cov_ext=zero, param=TERRAIN_PARAM,
                   state_lbs=STATE_LBS, state_ubs=STATE_UBS, input_lbs=INPUT_LBS, input_ubs=INPUT_UBS)
    results = {"true": run(env_true, 0.0, f"MPC_true_{name}")}
    for b in BETAS:
        env_l = Env(CASE_REAL, X0, XT, symbolic_h_mean_ext=h_gp, symbolic_h_cov_ext=s2h, symbolic_dh_cov_ext=s2dh,
                    param=TERRAIN_PARAM, state_lbs=STATE_LBS, state_ubs=STATE_UBS, input_lbs=INPUT_LBS, input_ubs=INPUT_UBS)
        results[f"b{b}"] = run(env_l, float(b), f"GPMPC_{name}_beta{b}")
        r = results[f"b{b}"]
        print(f"[{name}] beta={b}: violation {r['violation']:.4f}  cost {r['cost']:.1f}  "
              f"final |p - p_goal| {abs(r['states'][-1, 0] - XT[0]):.4f}  cap saturations {r['nsat']}  max-iter steps {r['nmaxiter']}")

    save = dict(p_train=p_train.ravel(), h_train=h_train.ravel(), p_grid=p_grid, h_true=h_true, gp_mean=gp_mean, gp_std=gp_std,
                v_max=V_MAX, dt=1 / FREQ, N=N, gap=np.array(GAP), target=XT[0], l_opt=l_opt, sf_opt=sf_opt)
    for k, r in results.items():
        for f in ("states", "inputs", "sigma", "pred", "Sigma_full_k0", "cost", "violation", "nsat", "nmaxiter"):
            save[f"{k}_{f}"] = r[f]
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, f"{name}.npz"), **save)
    return save


_SIMULATED = set()   # cases simulated in this process, so --rerun does each at most once


def load_case(name, rerun=False):
    """Cached results of a case, simulating it first if needed."""
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if (rerun and name not in _SIMULATED) or not os.path.exists(path):
        simulate_case(name)
        _SIMULATED.add(name)
    return dict(np.load(path))


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
PAPER_RC = {"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "font.family": "serif",
            "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42}
RAMP = ["#f0a08c", "#dd6b52", "#c0392b", "#7b241c"]      # beta = 0, 1, 2, 3
C_MAIN, C_TIGHT = RAMP[2], "tab:blue"
CAP = 0.8 * V_MAX                                        # MIN_WIDTH_FRACTION = 0.2 in GPMPCController
CAR_L, CAR_R = 0.2, 0.035                                # the car of utils/simulator.py, in data units
CAR_AXLE = 1.5 * CAR_R
CAR_COLOR = "steelblue"

h_true_at = lambda p: TERRAIN_PARAM * np.cos(18 * p)
slope_true = lambda p: -TERRAIN_PARAM * 18 * np.sin(18 * p)


def car_polygons(p):
    """Jeep outline and wheel centres (utils/simulator.add_car / animate), placed on the true terrain."""
    L, r = CAR_L, CAR_R
    th = np.arctan(slope_true(p))
    wheels = []
    for d in (-(L / 2 - r), (L / 2 - r)):
        pw = p + d * np.cos(th)
        thw = np.arctan(slope_true(pw))
        wheels.append((pw - r * np.sin(thw), h_true_at(pw) + r * np.cos(thw)))
    (w1x, w1y), (w2x, w2y) = wheels
    th_real = np.arctan2(w2y - w1y, w2x - w1x)
    centre = np.array([(w1x + w2x) / 2 - CAR_AXLE * np.sin(th_real), (w1y + w2y) / 2 + CAR_AXLE * np.cos(th_real)])
    shape = np.array([[-3 * L / 4 + L / 8, -r], [-2 * L / 3 + L / 8, L / 2 - r], [L / 4 - L / 8, L / 2 - r],
                      [L / 2 - L / 8, L / 4 - r], [3 * L / 4 - L / 8, L / 4 - r], [4 * L / 5 - L / 8, -r]])
    Rm = np.array([[np.cos(th_real), -np.sin(th_real)], [np.sin(th_real), np.cos(th_real)]])
    return shape @ Rm.T + centre, wheels


def draw_car(ax, p, filled=True):
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Polygon, Circle
    tint = lambda c, k: tuple(1.0 - k * (1.0 - np.array(to_rgb(c))))     # opaque blend towards white
    body, wheels = car_polygons(p)
    body_c = CAR_COLOR if filled else tint(CAR_COLOR, 0.4)               # the predicted car is a fainter copy
    wheel_c = "0.1" if filled else tint("0.1", 0.4)
    ax.add_patch(Polygon(body, closed=True, facecolor=body_c, edgecolor=body_c, lw=0.8, zorder=10))
    for w in wheels:
        ax.add_patch(Circle(w, CAR_R, facecolor=wheel_c, edgecolor=wheel_c, lw=0.7, zorder=11))


def panel_scene(ax, D):
    """True-aspect scene in the style of the course's Case 3 plot: car at the start, faded car at the
    position the t = 0 plan predicts at the end of the horizon, target flag, no-data gap shaded."""
    from matplotlib.patches import Polygon
    gap, target = D["gap"], float(D["target"])
    xlim = (-0.9, 0.9)
    box = ax.get_position()
    fig = ax.figure
    w_in, h_in = box.width * fig.get_figwidth(), box.height * fig.get_figheight()
    yr = (xlim[1] - xlim[0]) * h_in / w_in
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.40 * yr, 0.60 * yr)
    ax.set_aspect("equal", adjustable="box")
    ax.set_autoscale_on(False)
    ax.axvspan(gap[0], gap[1], color="0.93", zorder=0, lw=0)
    ax.fill_between(D["p_grid"], -1.0, D["h_true"], color="0.82", zorder=1, lw=0)
    ax.plot(D["p_grid"], D["h_true"], color="k", lw=1.0, zorder=2)
    p0 = D["b2_states"][0, 0]
    draw_car(ax, p0, filled=True)
    ax.annotate("start, $t=0$", (p0, h_true_at(p0) + CAR_L / 2 + CAR_AXLE), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=6)
    p_end = D["b2_pred"][0][-1, 0]
    draw_car(ax, p_end, filled=False)
    ax.annotate(f"predicted at $i={int(D['N'])}$", (p_end, h_true_at(p_end) + CAR_L / 2 + CAR_AXLE), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=6)
    ht, pole = h_true_at(target), 0.20
    ax.plot(target, ht, marker="x", color=C_MAIN, ms=5, mew=1.4, ls="none", zorder=12)
    ax.plot([target, target], [ht, ht + pole], color="k", lw=1.0, zorder=12)
    ax.add_patch(Polygon([[target, ht + pole], [target, ht + pole - 0.055], [target + 0.09, ht + pole - 0.0275]],
                         closed=True, facecolor=C_MAIN, edgecolor="none", zorder=12))
    ax.annotate("target", (target + 0.09, ht + pole - 0.0275), xytext=(3, 0), textcoords="offset points", ha="left", va="center", fontsize=6)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([0, 0.2])
    ax.set_ylabel("height $h$")
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_title("Problem Setup", pad=3)


def panel_profile(ax, D):
    """Strip under the scene, sharing its x-axis: terrain with ~10x vertical zoom and the learned GP."""
    gap = D["gap"]
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.031, 0.031)
    ax.set_autoscale_on(False)
    ax.axvspan(gap[0], gap[1], color="0.93", zorder=0, lw=0)
    ax.fill_between(D["p_grid"], D["gp_mean"] - 2 * D["gp_std"], D["gp_mean"] + 2 * D["gp_std"], color=C_MAIN, alpha=0.18, lw=0, zorder=1)
    ax.plot(D["p_grid"], D["gp_mean"], "--", color=C_MAIN, lw=0.8, zorder=2)
    ax.plot(D["p_grid"], D["h_true"], "-", color="k", lw=0.9, zorder=3)
    ax.text(-0.87, 0.020, "terrain $h(p)$", fontsize=5.5, ha="left", va="center")
    ax.text(0.0, 0.024, r"learned $\hat h(p)\pm2\sigma_h$", fontsize=5.5, ha="center", va="center", color=C_MAIN)
    for p in (D["b2_states"][0, 0], D["b2_pred"][0][-1, 0]):
        ax.axvline(p, color="0.5", lw=0.5, ls=":", zorder=4)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.02, 0, 0.02])
    ax.set_xlabel("position $p$", labelpad=1)
    ax.set_ylabel(r"$h$ (zoom)")


def draw_plan_ellipses(ax, mu, S, beta, **style):
    from matplotlib.patches import Ellipse
    for i in range(1, len(mu)):
        w, V = np.linalg.eigh(S[i])
        w = np.clip(w, 0.0, None)
        ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
        ax.add_patch(Ellipse(mu[i], 2 * beta * np.sqrt(w[0]), 2 * beta * np.sqrt(w[1]), angle=ang, **style))


def panel_statespace(ax, D, D_overlay=None, beta=2):
    """The t = 0 plan in the (p, v) plane with its 2-sigma covariance ellipses and the tightened limit."""
    from matplotlib.patches import Patch
    v_max = float(D["v_max"])
    mu, S = D["b2_pred"][0], D["b2_Sigma_full_k0"]
    sig_v = np.sqrt(S[:, 1, 1])
    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(-0.30, 0.92)
    ax.axhline(v_max, color="k", ls="--", lw=0.9, zorder=3, label=r"limit $v_{\max}$")
    if D_overlay is not None:   # the same plan under another covariance propagation, in faint gray
        mu2, S2 = D_overlay["b2_pred"][0], D_overlay["b2_Sigma_full_k0"]
        draw_plan_ellipses(ax, mu2, S2, beta, facecolor="0.94", edgecolor="0.84", lw=0.3, zorder=0.3)
        ax.plot(mu2[:, 0], mu2[:, 1], "-o", color="0.78", lw=0.9, ms=1.6, zorder=0.6)
        ax.text(mu2[-1, 0] + 0.03, mu2[-1, 1] - 0.06, "no ancillary", fontsize=5.5, color="0.7", ha="left", va="center", zorder=0.7)
    draw_plan_ellipses(ax, mu, S, beta, facecolor=C_MAIN, alpha=0.09, edgecolor="none", zorder=1)
    draw_plan_ellipses(ax, mu, S, beta, facecolor="none", edgecolor=C_MAIN, alpha=0.45, lw=0.4, zorder=2)
    ax.plot(mu[:, 0], mu[:, 1], "-o", color=C_MAIN, lw=1.2, ms=2.0, zorder=4, label=r"plan $\mu_{i|0}$")
    ax.plot(mu[1:, 0], v_max - np.minimum(beta * sig_v[1:], CAP), color=C_TIGHT, ls="-.", lw=1.0, zorder=5, label="tightened limit")
    ax.plot(mu[0, 0], mu[0, 1], "o", color="k", ms=3.2, zorder=6)
    ax.annotate("start", (mu[0, 0], mu[0, 1]), xytext=(4, -9), textcoords="offset points", fontsize=5.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(2, Patch(facecolor=C_MAIN, alpha=0.25, edgecolor=C_MAIN, lw=0.4))
    labels.insert(2, r"$2\sigma$ ellipses $\Sigma^x_{i|0}$")
    ax.legend(handles, labels, loc="upper left", ncol=2, frameon=False, fontsize=5, handlelength=1.2, columnspacing=0.6,
              handletextpad=0.3, labelspacing=0.3, borderaxespad=0.2)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel("position $p$", labelpad=1)
    ax.set_ylabel("velocity $v$")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_title(rf"Plan at $t = 0$ ($\beta = {beta}$)", pad=3)


def cost_formula(cfg):
    qp, qv = cfg["Q"]
    R = cfg["R"]
    if qv == 0:
        state = rf"{qp:g}(p_k-p_{{\mathrm{{goal}}}})^2"
    elif qv == qp:
        state = rf"{qp:g}\|x_k-x_{{\mathrm{{goal}}}}\|^2"
    else:
        state = rf"{qp:g}(p_k-p_{{\mathrm{{goal}}}})^2+{qv:g}v_k^2"
    r_txt = r"10^{%d}" % round(np.log10(R)) if R < 1e-3 else f"{R:g}"
    return rf"$J=\Sigma_k {state}+{r_txt}u_k^2$"


def panel_tradeoff(ax, D, cfg):
    from matplotlib.ticker import MaxNLocator
    viol = [float(D[f"b{b}_violation"]) for b in BETAS]
    cost = [float(D[f"b{b}_cost"]) for b in BETAS]
    ax.plot(viol, cost, "-", color="0.6", lw=0.8, zorder=1)
    for b, c, v, J in zip(BETAS, RAMP, viol, cost):
        ax.plot(v, J, "o", color=c, ms=4, zorder=2)
        ax.annotate(rf"$\beta={b}$", (v, J), textcoords="offset points", xytext=(4, 3), ha="left", fontsize=6, color=c)
    ax.axvline(0.0, color="k", ls=":", lw=0.7)
    ax.set_xlim(-0.03, 0.225)
    if cfg["cost_ylim"] is None:
        span = max(cost) - min(cost)
        ax.set_ylim(min(cost) - 0.08 * span - 2, max(cost) + 0.2 * span + 3)
    else:
        ax.set_ylim(*cfg["cost_ylim"])
    ax.set_xticks([0, 0.1, 0.2])
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.set_xlabel("constraint violation\n" + r"$\max(|v| - v_{\max}, 0)$", labelpad=1)
    ax.set_ylabel("closed-loop cost\n" + cost_formula(cfg), fontsize=6, labelpad=3)
    ax.set_title("Performance vs. Safety", pad=3)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def make_figure(name, rerun=False):
    """Build one case's figure (PNG and PDF in ex6_SysID/figures). Safe to call from the notebook."""
    import matplotlib.pyplot as plt

    cfg = CASES[name]
    D = load_case(cfg.get("data_from", name), rerun)
    D_overlay = load_case(cfg["overlay"], rerun) if "overlay" in cfg else None
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(3.5, 3.8))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.2], hspace=0.42)
        top = outer[0].subgridspec(2, 1, height_ratios=[1.0, 0.55], hspace=0.12)
        ax_scene = fig.add_subplot(top[0])
        ax_prof = fig.add_subplot(top[1], sharex=ax_scene)
        panel_scene(ax_scene, D)
        panel_profile(ax_prof, D)
        bot = outer[1].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.6)
        panel_statespace(fig.add_subplot(bot[0]), D, D_overlay)
        panel_tradeoff(fig.add_subplot(bot[1]), D, cfg)
        os.makedirs(FIG_DIR, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG_DIR, f"{cfg['output']}.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[{name}] wrote {os.path.join(FIG_DIR, cfg['output'])}.png/.pdf")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=sorted(CASES), action="append", help="case to build (repeatable)")
    ap.add_argument("--all", action="store_true", help="build every case")
    ap.add_argument("--rerun", action="store_true", help="re-simulate even if cached data exists")
    args = ap.parse_args()
    names = list(CASES) if args.all else (args.case or ["baseline"])
    for n in names:
        make_figure(n, rerun=args.rerun)
