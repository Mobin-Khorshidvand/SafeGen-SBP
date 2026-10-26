# SafeGen-SBP (v2.2.0)

**Author:** Mobin Khorshidvand  
**License:** MIT License  
**Environment:** Python 3.12.3 | NumPy 2.1.3 | SciPy 1.14.1  
**Status:** Verified (8/8 Unit & Integration Tests Passing | Reproducible Benchmarks)


## 💡 Key Features & Highlights
* **High Precision:** Implements robust numerical methods for precise half-space projections.
* **Modular Architecture:** Clean code structure allowing easy extension of safety bounds.
* **Reproducibility:** Complete documentation and verification logs included to replicate results effortlessly.


A compact Python engineering implementation of Euclidean projection onto a single linear half-space and intersections of linear half-spaces. It demonstrates exact single-constraint projection, Dykstra's classical cyclic projection algorithm, independent post-hoc feasibility checks, and paired benchmark comparisons with SciPy's SLSQP solver. This repository is an implementation and numerical verification package—not a novel optimization algorithm and not a proof of long-term closed-loop safety.

## Scope and limitations

For a single linear inequality `g.T @ x <= beta`, an already feasible vector is returned bitwise unchanged; a violating vector is projected orthogonally onto the boundary hyperplane. The projection mapping is **piecewise affine**: it acts as the identity map on the feasible half-space and an affine projection on the violating region, remaining continuous everywhere though generally non-differentiable on the boundary.

For multiple linear constraints `G @ x <= b`, the package uses Dykstra's cyclic algorithm. Step convergence (`converged`), iteration-budget exhaustion (`budget_exhausted`), and constraint satisfaction (`is_feasible`) are reported as separate states. Failing to certify a feasible point within a finite iteration budget does not constitute a formal certificate of geometric infeasibility. All experiments operate on synthetic Gaussian vectors and static linear constraints in `float64` arithmetic; there are no hardware, robot-trajectory, nonlinear-obstacle, or closed-loop stability experiments.

## Recorded verification environment

The included verification log (`results/final_verification.log`) was generated using **Python 3.12.3**, **NumPy 2.1.3**, and **SciPy 1.14.1**. Numerical residuals and wall-clock timings can vary across hardware, operating systems, and BLAS builds.

## Installation and execution

From a clean environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python safegen_sbp.py
```

The script runs all eight asserted verification tests followed by both reproducible benchmarks. To regenerate and save the complete verification log:

```bash
python safegen_sbp.py | tee results/final_verification.log
```

## Key verified results (v2.2.0)

- **Test 1 (2D analytical example):** Projecting `[4, 5]` onto `x1 + x2 <= 4` returns `[1.5, 2.5]`.
- **Test 3 & Test 4 (Numerical policy & denominator offset):** Enforces a squared-norm cutoff `g.T @ g <= 1e-14` (`||g||_2 = 1e-6` accepted, `||g||_2 = 1e-8` rejected), explicitly rejects zero/`NaN`/`Inf` normals, incompatible dimensions, and invalid tolerances/budgets, and verifies that adding `1e-12` to the denominator halves the corrective step (`[2, 5] -> [1.5, 5]` instead of `[1, 5]`).
- **Test 5 (Large-scale 2D normal):** Projects `[4, 3]` onto `1e6 * x1 <= 2e6` accurately to `[2, 3]`.
- **Test 6 (SLSQP agreement):** With `step_tol=1e-12` and `feas_tol=1e-12`, Dykstra matches SciPy SLSQP to an `L_inf` gap of `1.152855588771e-12` (asserted `< 1.1e-9`).
- **Test 7 & Test 8 (Inconsistent constraints & budget cap):** On contradictory bounds `x1 + x2 <= 2` and `-x1 - x2 <= -5`, 50 cycles yield a maximum signed violation of `3.0` (`is_feasible=False`, `x=None`) without claiming an infeasibility certificate.
- **Benchmark 1 (50D single half-space, 5 seeds × 1,000 samples):** Raw violation rate is `52.200000% +/- 20.801250%` (mean of per-seed max violations `3.334013636005e-06`), replacing an unverified `74.72% +/- 0.83%` figure from an earlier draft. Exact projection achieves `0.000000%` violations (mean per-seed max residual `1.072201025912e-21`).
- **Benchmark 2 (20D polytope `sum(x) <= 15`, `-x_i <= 0`, 5 seeds × 200 samples):** Tracked independently, both Dykstra (`1000/1000`, max violation `9.999362582391e-08`, mean cycles `27.945000`) and SLSQP (`1000/1000`, max violation `1.044497821567e-12`) succeeded on all 1,000 samples with a maximum `L_inf` solution gap of `1.632513857430e-08`. Single-pass mean times in this environment were `13.620408 ms/sample` (Dykstra) and `0.475233 ms/sample` (SLSQP), giving a ratio of `0.034891x` (SLSQP/Dykstra).

## Repository structure

- `safegen_sbp.py`: Executable implementation, unit tests, and benchmarks (v2.2.0).
- `requirements.txt`: Pinned dependencies (`numpy==2.1.3`, `scipy==1.14.1`).
- `results/final_verification.log`: Complete verified stdout from the final run.
- `docs/technical_report.pdf` & `docs/technical_report.md`: Five-page technical report.
- `docs/technical_report.tex`: LaTeX source used to compile `docs/technical_report.pdf`.
- `docs/reproducibility_guide.pdf` & `docs/reproducibility_guide.md`: Environment and reproduction instructions.
- `docs/verification_audit.pdf` & `docs/verification_audit.md`: Audit trail and discrepancy record.
- `docs/oral_defense_guide.pdf` & `docs/oral_defense_guide.md`: Concise Q&A study guide.
- `docs/license_decision.md`: Copyright and licensing note prior to public release.

## License status

No license grant was supplied in the original project materials. See `docs/license_decision.md`: MIT is recommended if permissive open-source release is desired, but no license is asserted until confirmed by the copyright holder.


---

## 🤝 Contributing & Contact
Contributions, suggestions, and feedback are always welcome! Feel free to open an issue or submit a pull request.

* **Author:** Mobin Khorshidvand[cite: 3]
* **GitHub:** [Mobin-Khorshidvand](https://github.com/Mobin-Khorshidvand)
* **LinkedIn:** [Profile Link](https://www.linkedin.com/in/mobin-khorshidvand-50459122a)

⭐ If you found this project useful, please consider giving it a star on GitHub!

