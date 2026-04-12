"""
ukf.py
======
Dual Unscented Kalman Filter (UKF) for joint state and parameter estimation.

Mirrors the two-pass structure of EnKF_ParameterEstimation exactly:
  Pass 1 (measurement steps only): sigma points in parameter space → parameter update
  Pass 2 (every step):             sigma points in state space     → state propagation + update

Sigma point scheme: Merwe scaled (Van der Merwe, 2004).

Default hyperparameters
-----------------------
  alpha = 0.5   — spread of sigma points around the mean; small values (1e-3)
                  cause large negative Wc[0] that destabilises covariance when
                  state scales span many orders of magnitude
  beta  = 2.0   — encodes prior knowledge of distribution; 2.0 is optimal
                  for Gaussian distributions
  kappa = 0.0   — secondary scaling; 0.0 is the standard choice

For an n-dimensional space these give 2n+1 sigma points:
  Parameter pass (n=24): 49 sigma points
  State pass     (n=8):  17 sigma points
"""

import numpy as np
from numpy.linalg import inv, cholesky

from cho_enkf.model import model_step


class UKF_DualParameterEstimation:
    """
    Dual UKF for joint state and parameter estimation.

    Attributes set externally before running (same interface as EnKF):
      Q, R, H   — noise/observation matrices
      P_x       — initial state covariance  (8×8)
      P_theta   — initial parameter covariance (n_p×n_p)
    """

    def __init__(self, dataset_name: str, datasets: dict,
                 dataset_noise_variances: dict, dt_model: float,
                 mean_parameters: dict,
                 alpha: float = 0.5, beta: float = 2.0, kappa: float = 0.0):
        self.dataset_name = dataset_name
        self.dt    = dt_model
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa

        exp_meas   = datasets[dataset_name]["exp_meas"]
        state_cols = ['Xv', 'mAb', 'Glc', 'Amm', 'Gln', 'Lac', 'Glu', 'Asn']
        self.x     = exp_meas.iloc[0][state_cols].values.astype(float)
        self.z     = exp_meas[state_cols].values
        self.H     = np.eye(len(self.x))

        self.param_keys = list(mean_parameters.keys())
        self.theta      = mean_parameters.copy()
        self.theta_vec  = np.array(list(mean_parameters.values()), dtype=float)

        # Set by runner before the main loop
        self.Q       = None   # state process noise   (8×8)
        self.R       = None   # observation noise     (8×8)
        self.P_x     = None   # state covariance      (8×8)
        self.P_theta = None   # parameter covariance  (n_p×n_p)

        # Cache between forecast and update calls
        self._param_sigmas     = None
        self._param_Wm         = None
        self._param_Wc         = None
        self._param_propagated = None   # predicted observations (H=I, so = propagated states)
        self._param_z_mean     = None

        self._state_propagated = None
        self._state_Wm         = None
        self._state_Wc         = None
        self._state_x_pred     = None

        # Records
        self.para_records = [self.theta.copy()]

    # ── Sigma point generation ────────────────────────────────────────────────

    def _sigma_points(self, mean: np.ndarray, cov: np.ndarray):
        """
        Merwe scaled sigma points and weights for an n-dimensional distribution.

        Returns
        -------
        sigmas : (2n+1, n)
        Wm     : (2n+1,)  mean weights
        Wc     : (2n+1,)  covariance weights
        """
        n   = len(mean)
        lam = self.alpha**2 * (n + self.kappa) - n

        Wm    = np.full(2 * n + 1, 0.5 / (n + lam))
        Wc    = np.full(2 * n + 1, 0.5 / (n + lam))
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)

        M = (n + lam) * cov
        # Nearest positive-definite matrix via eigenvalue clipping —
        # more robust than a fixed jitter when covariance has drifted.
        M = 0.5 * (M + M.T)                          # ensure symmetry
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-10)          # floor negative eigenvalues
        M = eigvecs @ np.diag(eigvals) @ eigvecs.T
        try:
            L = cholesky(M)
        except np.linalg.LinAlgError:
            # Last resort: add a scaled identity
            L = cholesky(M + 1e-6 * np.eye(n))

        sigmas      = np.empty((2 * n + 1, n))
        sigmas[0]   = mean
        for i in range(n):
            sigmas[i + 1]     = mean + L[:, i]
            sigmas[n + i + 1] = mean - L[:, i]

        return sigmas, Wm, Wc

    # ── Pass 1: parameter forecast & update ──────────────────────────────────

    def forecast_for_parameters(self, controls):
        """
        Pass 1 forecast (called at measurement times only).

        Generates 49 sigma points in parameter space, propagates the current
        state mean through the model with each sigma point's parameters.
        Results are cached for parameters_update().
        """
        Fin_t, Fout_t, Fin_glc_t, V_t, feed = controls
        sigmas_theta, Wm, Wc = self._sigma_points(self.theta_vec, self.P_theta)

        # All model parameters must be positive — clip sigma points before
        # propagation to prevent ODE overflow from unphysical parameter values.
        sigmas_theta_safe = np.maximum(sigmas_theta, 1e-12)

        propagated = np.array([
            model_step(dict(zip(self.param_keys, sig)),
                       self.x, 0.0, Fin_t, Fout_t, Fin_glc_t, V_t, self.dt, feed)
            for sig in sigmas_theta_safe
        ])  # shape (49, 8)

        # Replace any NaN/Inf rows (blown-up ODE) with the current state mean
        # so they do not corrupt the cross-covariance.
        bad = ~np.isfinite(propagated).all(axis=1)
        propagated[bad] = self.x

        self._param_sigmas     = sigmas_theta_safe
        self._param_Wm         = Wm
        self._param_Wc         = Wc
        self._param_propagated = propagated
        self._param_z_mean     = Wm @ propagated   # H = I

    def parameters_update(self, time_index: int):
        """
        Pass 1 update: UKF parameter correction using the cached propagations.
        Updates theta_vec, theta (dict), and P_theta.
        """
        z_meas       = self.z[time_index]
        sigmas_theta = self._param_sigmas       # (49, n_p)
        propagated   = self._param_propagated   # (49, 8)
        z_mean       = self._param_z_mean       # (8,)
        Wm, Wc       = self._param_Wm, self._param_Wc
        n_p          = len(self.theta_vec)

        # Innovation covariance  Pzz = Σ Wc·(zⁱ-z̄)(zⁱ-z̄)ᵀ + R
        Pzz = (sum(Wc[i] * np.outer(propagated[i] - z_mean,
                                     propagated[i] - z_mean)
                   for i in range(len(Wm)))
               + self.R)

        # Cross-covariance  Pθz = Σ Wc·(θⁱ-θ̄)(zⁱ-z̄)ᵀ
        Pthz = sum(Wc[i] * np.outer(sigmas_theta[i] - self.theta_vec,
                                     propagated[i] - z_mean)
                   for i in range(len(Wm)))

        Kp             = Pthz @ inv(Pzz)
        self.theta_vec = np.maximum(self.theta_vec + Kp @ (z_meas - z_mean), 1e-12)

        # Joseph-like form: P - K Pzz K^T rewritten as
        # (I - K H_eff) P (I - K H_eff)^T + K R K^T
        # where H_eff = Pthz^T @ inv(P_theta) is the effective Jacobian.
        # Equivalent but more stable: use the standard form with eigenvalue floor.
        self.P_theta  = self.P_theta - Kp @ Pzz @ Kp.T
        self.P_theta  = 0.5 * (self.P_theta + self.P_theta.T)
        ev, evec      = np.linalg.eigh(self.P_theta)
        self.P_theta  = evec @ np.diag(np.maximum(ev, 1e-12)) @ evec.T

        self.theta = dict(zip(self.param_keys, self.theta_vec))
        self.para_records.append(self.theta.copy())

    # ── Pass 2: state forecast & update ──────────────────────────────────────

    def forecast_for_states(self, controls):
        """
        Pass 2 forecast (called at every time step).

        Generates 17 sigma points in state space, propagates each through the
        model with the current parameter mean. Updates self.x and self.P_x.
        Propagated sigma points are cached for states_update().
        """
        Fin_t, Fout_t, Fin_glc_t, V_t, feed = controls
        sigmas_x, Wm, Wc = self._sigma_points(self.x, self.P_x)

        propagated = np.array([
            model_step(self.theta, sig, 0.0, Fin_t, Fout_t, Fin_glc_t, V_t, self.dt, feed)
            for sig in sigmas_x
        ])  # shape (17, 8)

        # Replace any NaN/Inf rows with the current state mean as a fallback.
        bad = ~np.isfinite(propagated).all(axis=1)
        propagated[bad] = self.x

        x_pred = Wm @ propagated   # (8,)

        P_pred = (sum(Wc[i] * np.outer(propagated[i] - x_pred,
                                        propagated[i] - x_pred)
                      for i in range(len(Wm)))
                  + self.Q)

        self.x   = np.maximum(x_pred, 1e-12)
        self.P_x = P_pred

        self._state_propagated = propagated
        self._state_Wm         = Wm
        self._state_Wc         = Wc
        self._state_x_pred     = x_pred

    def states_update(self, time_index: int):
        """
        Pass 2 update: UKF state correction using the cached propagations.
        Updates self.x and self.P_x.
        """
        z_meas     = self.z[time_index]
        propagated = self._state_propagated   # (17, 8)
        x_pred     = self._state_x_pred       # (8,)
        Wm, Wc     = self._state_Wm, self._state_Wc

        # H = I → predicted observations equal propagated states
        z_mean = Wm @ propagated   # (8,)

        Pzz = (sum(Wc[i] * np.outer(propagated[i] - z_mean,
                                     propagated[i] - z_mean)
                   for i in range(len(Wm)))
               + self.R)

        Pxz = sum(Wc[i] * np.outer(propagated[i] - x_pred,
                                    propagated[i] - z_mean)
                  for i in range(len(Wm)))

        Ks       = Pxz @ inv(Pzz)
        self.x   = np.maximum(x_pred + Ks @ (z_meas - z_mean), 1e-12)

        # Joseph form: unconditionally preserves positive-definiteness
        IKH          = np.eye(len(self.x)) - Ks @ self.H
        self.P_x     = IKH @ self.P_x @ IKH.T + Ks @ self.R @ Ks.T
        self.P_x     = 0.5 * (self.P_x + self.P_x.T)
