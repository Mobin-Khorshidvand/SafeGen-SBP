"""SafeGen-SBP: Euclidean projection onto linear half-spaces and intersections.

This is an engineering implementation of classical projection methods, not a
claim of a new algorithm or a guarantee of long-term system safety.
"""
from __future__ import annotations

import platform
import time
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize

__version__ = "2.2.0"
DEFAULT_ZERO_NORM_SQ_TOL = 1e-14


def _vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    return arr


def project_single_halfspace(theta: Any, g: Any, beta: float,
                              zero_norm_sq_tol: float = DEFAULT_ZERO_NORM_SQ_TOL) -> np.ndarray:
    """Return Euclidean projection onto H={x: g.T x <= beta}.

    ``zero_norm_sq_tol`` is a numerical policy applied to ||g||_2**2, not a
    mathematical definition of zero. Normals at or below that cutoff are
    rejected because the projection is too sensitive to their scaling.
    """
    theta = _vector(theta, "theta")
    g = _vector(g, "g")
    if theta.shape != g.shape:
        raise ValueError("theta and g must have the same shape")
    if not np.isfinite(beta) or not np.isfinite(zero_norm_sq_tol) or zero_norm_sq_tol < 0:
        raise ValueError("beta and zero_norm_sq_tol must be finite; tolerance must be nonnegative")
    norm_sq = float(np.dot(g, g))
    if not np.isfinite(norm_sq) or norm_sq <= zero_norm_sq_tol:
        raise ValueError(f"normal rejected: squared norm {norm_sq:.6g} <= policy cutoff {zero_norm_sq_tol:.6g}")
    violation = float(np.dot(g, theta) - beta)
    if not np.isfinite(violation):
        raise ValueError("computed violation is not finite")
    if violation <= 0.0:
        return theta.copy()
    return theta - (violation / norm_sq) * g


def legacy_project_buggy(theta: Any, g: Any, beta: float, eps: float = 1e-12) -> np.ndarray:
    """Pedagogical reconstruction of projection with an additive denominator offset.

    It models a hypothetical formula; it does not establish that any external
    package or historical implementation contained this defect.
    """
    theta = _vector(theta, "theta")
    g = _vector(g, "g")
    if theta.shape != g.shape or not np.isfinite(beta) or not np.isfinite(eps) or eps < 0:
        raise ValueError("invalid dimensions, beta, or eps")
    norm_sq = float(np.dot(g, g))
    violation = float(np.dot(g, theta) - beta)
    if violation <= 0:
        return theta.copy()
    if norm_sq + eps <= 0 or not np.isfinite(norm_sq + eps):
        raise ValueError("invalid denominator")
    return theta - (violation / (norm_sq + eps)) * g


def _validate_problem(theta_init: Any, G: Any, b: Any, zero_norm_sq_tol: float):
    x0 = _vector(theta_init, "theta_init")
    G = np.asarray(G, dtype=np.float64)
    b = _vector(b, "b")
    if G.ndim != 2 or not np.all(np.isfinite(G)):
        raise ValueError("G must be a finite 2D matrix")
    m, n = G.shape
    if n != x0.size or b.shape != (m,) or m == 0:
        raise ValueError("incompatible or empty dimensions for theta_init, G, and b")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must be finite")
    if not np.isfinite(zero_norm_sq_tol) or zero_norm_sq_tol < 0:
        raise ValueError("zero_norm_sq_tol must be finite and nonnegative")
    norms_sq = np.einsum("ij,ij->i", G, G)
    bad = np.flatnonzero(~np.isfinite(norms_sq) | (norms_sq <= zero_norm_sq_tol))
    if bad.size:
        raise ValueError(f"constraint row(s) {bad.tolist()} rejected by squared-normal cutoff {zero_norm_sq_tol:g}")
    return x0, G, b


def dykstra_linear_projection(theta_init: Any, G: Any, b: Any, max_iters: int = 500,
                               step_tol: float = 1e-8, feas_tol: float = 1e-7,
                               zero_norm_sq_tol: float = DEFAULT_ZERO_NORM_SQ_TOL) -> dict:
    """Dykstra projection with distinct stationarity, budget, and feasibility states.

    ``converged`` means both step and feasibility tests met. ``budget_exhausted``
    means the iteration cap was consumed. ``is_feasible`` is an independent
    final check. No failure to find a feasible iterate is interpreted as proof
    that the intersection is empty.
    """
    x0, G, b = _validate_problem(theta_init, G, b, zero_norm_sq_tol)
    if isinstance(max_iters, bool) or not isinstance(max_iters, (int, np.integer)) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")
    if not np.isfinite(step_tol) or step_tol < 0 or not np.isfinite(feas_tol) or feas_tol < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    m, n = G.shape
    x = x0.copy()
    corrections = np.zeros((m, n), dtype=np.float64)
    converged = False
    step_diff = float("inf")
    iterations = 0
    for iterations in range(1, max_iters + 1):
        previous = x.copy()
        for i in range(m):
            y = x + corrections[i]
            projected = project_single_halfspace(y, G[i], b[i], zero_norm_sq_tol)
            corrections[i] = y - projected
            x = projected
        step_diff = float(np.max(np.abs(x - previous)))
        raw_violation = np.dot(G, x) - b
        max_signed_violation = float(np.max(raw_violation))
        if step_diff <= step_tol and max_signed_violation <= feas_tol:
            converged = True
            break
    raw_violation = np.dot(G, x) - b
    max_signed_violation = float(np.max(raw_violation))
    max_violation = max(0.0, max_signed_violation)
    is_feasible = max_signed_violation <= feas_tol
    return {
        "x": x.copy() if is_feasible else None,
        "x_raw": x.copy(),
        "is_feasible": bool(is_feasible),
        "converged": bool(converged),
        "budget_exhausted": bool(not converged and iterations >= max_iters),
        "step_diff_inf": step_diff,
        "max_violation": max_violation,
        "max_signed_violation": max_signed_violation,
        "iterations": iterations,
    }


def solve_scipy_slsqp(theta_init: Any, G: Any, b: Any, ftol: float = 1e-12,
                       maxiter: int = 500) -> dict:
    """Solve the same Euclidean projection QP using SciPy SLSQP."""
    x0 = _vector(theta_init, "theta_init")
    G = np.asarray(G, dtype=np.float64)
    b = _vector(b, "b")
    if G.ndim != 2 or G.shape != (b.size, x0.size) or not np.all(np.isfinite(G)) or not np.all(np.isfinite(b)):
        raise ValueError("invalid dimensions or non-finite problem data")
    res = minimize(fun=lambda x: 0.5 * np.dot(x - x0, x - x0), x0=x0,
                   jac=lambda x: x - x0,
                   constraints={"type": "ineq", "fun": lambda x: b - G @ x,
                                "jac": lambda x: -G}, method="SLSQP",
                   options={"ftol": ftol, "maxiter": maxiter})
    signed = G @ res.x - b
    return {"x": res.x, "max_violation": max(0.0, float(np.max(signed))),
            "max_signed_violation": float(np.max(signed)), "success": bool(res.success),
            "status": int(res.status), "message": str(res.message), "iterations": int(res.nit)}


def run_unit_tests() -> list[str]:
    passed = []
    p = project_single_halfspace([4., 5.], [1., 1.], 4.)
    assert np.allclose(p, [1.5, 2.5], atol=1e-12) and abs(np.sum(p)-4) <= 1e-12
    passed.append("Test 1: exact 2D projection [4,5] -> [1.5,2.5] and boundary")
    safe = project_single_halfspace([1., 2.], [1., 1.], 4.)
    assert np.array_equal(safe, [1., 2.])
    passed.append("Test 2: safe input remains bitwise unchanged")
    for normal in ([0., 0.], [np.nan, 1.], [np.inf, 1.], [-np.inf, 1.]):
        try:
            project_single_halfspace([1., 2.], normal, 4.)
            raise AssertionError("invalid normal was accepted")
        except ValueError:
            pass
    for bad_call in (
        lambda: project_single_halfspace([1., 2., 3.], [1., 1.], 4.),
        lambda: project_single_halfspace([1., 2.], [1., 1.], 4., zero_norm_sq_tol=-1e-14),
        lambda: dykstra_linear_projection([1., 2.], [[1., 1., 1.]], [1.]),
        lambda: dykstra_linear_projection([1., 2.], [[1., 1.]], [1., 2.]),
        lambda: dykstra_linear_projection([1., 2.], [[1., 1.]], [1.], max_iters=0),
        lambda: dykstra_linear_projection([1., 2.], [[1., 1.]], [1.], step_tol=-1e-8),
        lambda: dykstra_linear_projection([1., 2.], [[1., 1.]], [1.], feas_tol=np.nan),
    ):
        try:
            bad_call()
            raise AssertionError("invalid dimensions, tolerances, or iteration budget accepted")
        except ValueError:
            pass
    passed.append("Test 3: zero/NaN/Inf normals, incompatible dimensions, and invalid tolerances/budgets rejected explicitly")
    tiny = project_single_halfspace([2., 5.], [1e-6, 0.], 1e-6)
    assert np.allclose(tiny, [1., 5.], atol=1e-12)
    offset = legacy_project_buggy([2., 5.], [1e-6, 0.], 1e-6, eps=1e-12)
    assert np.allclose(offset, [1.5, 5.], atol=1e-12)
    try:
        project_single_halfspace([1., 2.], [1e-8, 0.], 0.)
        raise AssertionError("normal below squared-norm cutoff accepted")
    except ValueError:
        pass
    passed.append("Test 4: valid 1e-6 normal works; 1e-8 normal rejected by documented squared-norm cutoff")
    large = project_single_halfspace([4., 3.], [1e6, 0.], 2e6)
    assert np.allclose(large, [2., 3.], atol=1e-12)
    passed.append("Test 5: large-scale 2D normal projects [4,3] to [2,3]")
    G = np.array([[1., 2.], [2., 1.], [-1., 0.], [0., -1.]])
    b = np.array([2., 2., 0., 0.])
    d = dykstra_linear_projection([3., 3.], G, b, step_tol=1e-12, feas_tol=1e-12)
    s = solve_scipy_slsqp([3., 3.], G, b)
    gap = float(np.max(np.abs(d["x"]-s["x"])))
    assert d["is_feasible"] and s["success"] and gap < 1.1e-9
    passed.append(f"Test 6: Dykstra feasible; SLSQP success; L_inf solution gap={gap:.12e} (<1.1e-9)")
    Gi = np.array([[1., 1.], [-1., -1.]])
    bi = np.array([2., -5.])
    ri = dykstra_linear_projection([0., 0.], Gi, bi, max_iters=50)
    assert not ri["is_feasible"] and ri["x"] is None and ri["budget_exhausted"]
    passed.append(f"Test 7: contradictory example not certified; 50 cycles, signed max violation={ri['max_signed_violation']:.12g}; no infeasibility claim")
    cap = dykstra_linear_projection([10., 10.], G, b, max_iters=1, step_tol=1e-15)
    assert cap["budget_exhausted"] and not cap["converged"] and cap["x"] is None
    passed.append(f"Test 8: one-cycle cap distinct from safety; converged={cap['converged']}, feasible={cap['is_feasible']}, violation={cap['max_violation']:.12g}")
    return passed


def run_benchmarks() -> list[str]:
    lines = []
    seeds = [42, 43, 44, 45, 46]
    lines.append(f"Environment: Python {platform.python_version()} | NumPy {np.__version__} | SciPy {scipy.__version__}")
    lines.append(f"Seeds: {seeds}; RNG: numpy.random.default_rng(seed)")
    raw_rates, raw_max, legacy_rates, legacy_max, legacy_gaps, exact_rates, exact_max = [], [], [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        g = rng.standard_normal(50)
        g = g / np.linalg.norm(g) * 1e-6
        points = rng.normal(0.5, 1.0, (1000, 50))
        raw_v = points @ g
        exact = np.array([project_single_halfspace(x, g, 0.) for x in points])
        legacy = np.array([legacy_project_buggy(x, g, 0.) for x in points])
        exact_v = exact @ g
        legacy_v = legacy @ g
        raw_rates.append(float(np.mean(raw_v > 0)))
        raw_max.append(float(np.max(raw_v)))
        legacy_rates.append(float(np.mean(legacy_v > 1e-15)))
        legacy_max.append(float(max(0., np.max(legacy_v))))
        legacy_gaps.append(float(np.max(np.linalg.norm(legacy-exact, axis=1))))
        exact_rates.append(float(np.mean(exact_v > 1e-12)))
        exact_max.append(float(max(0., np.max(exact_v))))
    mean_sd = lambda xs: (float(np.mean(xs)), float(np.std(xs, ddof=0)))
    lines.append("Benchmark 1: n=50, 5 seeds x 1000 samples=5000; ||g||2=1e-6; beta=0; theta~N(0.5,1.0). Rates shown mean +/- population SD of five per-seed rates.")
    for label, rates, maxima in [("raw",raw_rates,raw_max),("legacy +1e-12 (reconstructed)",legacy_rates,legacy_max),("exact",exact_rates,exact_max)]:
        mu, sd = mean_sd(rates)
        lines.append(f"  {label}: violation_rate={mu*100:.6f}% +/- {sd*100:.6f}%; mean_of_per_seed_max_violation={np.mean(maxima):.12e}")
    lines.append(f"  legacy: mean of per-seed max Euclidean gaps to exact={np.mean(legacy_gaps):.12e}")

    G = np.vstack([np.ones((1,20)), -np.eye(20)])
    b = np.concatenate([[15.], np.zeros(20)])
    d_times = []
    s_times = []
    d_viol = []
    s_viol = []
    gaps = []
    iters = []
    d_ok = 0
    s_ok = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        points = rng.normal(1., 2., (200, 20))
        for x0 in points:
            d0 = time.perf_counter()
            d = dykstra_linear_projection(x0, G, b)
            d_times.append(time.perf_counter() - d0)
            s0 = time.perf_counter()
            s = solve_scipy_slsqp(x0, G, b)
            s_times.append(time.perf_counter() - s0)
            d_viol.append(d["max_violation"])
            s_viol.append(s["max_violation"])
            iters.append(d["iterations"])
            if d["is_feasible"]:
                d_ok += 1
            if s["success"]:
                s_ok += 1
            if d["is_feasible"] and s["success"]:
                gaps.append(float(np.max(np.abs(d["x"] - s["x"]))))
    dt = float(np.mean(d_times))
    st = float(np.mean(s_times))
    lines.append("Benchmark 2: n=20, m=21; constraints sum(x)<=15 and -x_i<=0; 5 seeds x 200=1000 identical inputs; theta~N(1,2); Dykstra step_tol=1e-8, feas_tol=1e-7, max_iters=500; SLSQP ftol=1e-12, maxiter=500.")
    lines.append("  Timing: sequential per-sample time.perf_counter, one pass (no warm-up/repeat); timing is environment-specific.")
    lines.append(f"  Dykstra certified feasible: {d_ok}/1000; max violation={max(d_viol):.12e}; mean iterations={np.mean(iters):.6f}; mean time={dt*1e3:.6f} ms/sample")
    lines.append(f"  SLSQP success: {s_ok}/1000; max violation={max(s_viol):.12e}; mean time={st*1e3:.6f} ms/sample")
    lines.append(f"  max L_inf solution gap on jointly successful cases={max(gaps):.12e}" if gaps else "  no jointly successful cases")
    lines.append(f"  empirical ratio SLSQP/Dykstra={st/dt:.6f}x (not a general performance claim)")
    return lines


def main() -> None:
    print("="*72)
    print("SAFEGEN-SBP: 8 VERIFICATION TESTS")
    print("="*72)
    for line in run_unit_tests():
        print("[PASS] " + line)
    print("="*72)
    print("SAFEGEN-SBP: REPRODUCIBLE BENCHMARKS")
    print("="*72)
    for line in run_benchmarks():
        print(line)
    print("="*72)


if __name__ == "__main__":
    main()
