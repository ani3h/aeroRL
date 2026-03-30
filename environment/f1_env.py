"""
F1 Aerodynamics Gymnasium Environment

Custom Gym environment that simulates F1 car aerodynamic optimization.
Uses an analytical surrogate CFD model for fast RL training.
The trained policy can later be validated against real CFD simulations.
"""

import os
import logging
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class F1AeroEnv(gym.Env):
    """
    Gymnasium environment for F1 aerodynamic parameter optimization.

    Observation Space (8-dim continuous):
        0: drag_coefficient (Cd)
        1: downforce_coefficient (-Cl, positive = more downforce)
        2: efficiency (downforce / drag)
        3: pressure_recovery
        4: flow_velocity_ratio
        5: flow_separation_index
        6: turbulence_intensity
        7: surface_friction_coef

    Action Space (8-dim continuous, each in [-1, 1]):
        Mapped to geometry parameter deltas:
        0: front_wing_angle delta
        1: rear_wing_angle delta
        2: diffuser_angle delta
        3: ride_height delta
        4: nose_height delta
        5: side_pod_shape delta
        6: floor_edge_height delta
        7: gurney_flap_size delta
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # Parameter definitions: (name, min, max, default, max_delta_per_step)
    PARAM_DEFS = [
        ("front_wing_angle", 0.0, 15.0, 7.5, 2.0),
        ("rear_wing_angle", 5.0, 25.0, 15.0, 2.0),
        ("diffuser_angle", 5.0, 15.0, 10.0, 1.5),
        ("ride_height", 50.0, 120.0, 80.0, 10.0),
        ("nose_height", 100.0, 250.0, 175.0, 15.0),
        ("side_pod_shape", 0.0, 5.0, 2.5, 0.5),
        ("floor_edge_height", 10.0, 50.0, 30.0, 5.0),
        ("gurney_flap_size", 0.0, 15.0, 5.0, 2.0),
    ]

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        # Load config if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        # Reward weights from config (with defaults)
        reward_cfg = self.config.get("environment", {}).get("rewards", {})

        # Primary: efficiency
        self.efficiency_target = reward_cfg.get("efficiency_target", 1.5)
        self.efficiency_weight = reward_cfg.get("efficiency_weight", 5.0)

        # Secondary: drag & downforce
        self.drag_penalty = reward_cfg.get("drag_penalty", -2.0)
        self.drag_baseline = reward_cfg.get("drag_baseline", 0.32)
        self.downforce_weight = reward_cfg.get("downforce_weight", 1.5)
        self.downforce_baseline = reward_cfg.get("downforce_baseline", 0.35)

        # Improvement bonuses
        self.efficiency_improvement_bonus = reward_cfg.get("efficiency_improvement_bonus", 3.0)
        self.new_best_bonus = reward_cfg.get("new_best_bonus", 1.0)

        # Penalties
        self.smoothness_penalty = reward_cfg.get("smoothness_penalty", -0.5)
        self.illegal_state_penalty = reward_cfg.get("illegal_state_penalty", -2.0)

        # Build parameter arrays
        self.param_names = [p[0] for p in self.PARAM_DEFS]
        self.param_min = np.array([p[1] for p in self.PARAM_DEFS], dtype=np.float32)
        self.param_max = np.array([p[2] for p in self.PARAM_DEFS], dtype=np.float32)
        self.param_default = np.array([p[3] for p in self.PARAM_DEFS], dtype=np.float32)
        self.param_delta_max = np.array([p[4] for p in self.PARAM_DEFS], dtype=np.float32)
        self.n_params = len(self.PARAM_DEFS)

        # --- Spaces ---
        # Actions are continuous in [-1, 1], scaled to deltas
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32
        )

        # Observations: 8 aerodynamic metrics (all normalised roughly to [0, ~5])
        obs_low = np.zeros(8, dtype=np.float32)
        obs_high = np.full(8, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Internal state
        self.current_params = None
        self.current_step = 0
        self.best_efficiency = 0.0
        self.prev_efficiency = 0.0
        self.prev_action = np.zeros(self.n_params, dtype=np.float32)

        logger.info("F1AeroEnv initialized")

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Start from default or slightly randomised parameters
        if options and options.get("random_start", False):
            self.current_params = self.np_random.uniform(
                self.param_min, self.param_max
            ).astype(np.float32)
        else:
            # Small perturbation around defaults
            noise = self.np_random.uniform(-0.1, 0.1, size=self.n_params).astype(np.float32)
            self.current_params = self.param_default + noise * (self.param_max - self.param_min)
            self.current_params = np.clip(self.current_params, self.param_min, self.param_max)

        self.current_step = 0
        self.best_efficiency = 0.0
        self.prev_action = np.zeros(self.n_params, dtype=np.float32)

        obs = self._compute_observation()
        self.prev_efficiency = obs[2]
        self.best_efficiency = obs[2]
        info = self._build_info(obs)

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).flatten()
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Map action [-1, 1] → parameter deltas
        deltas = action * self.param_delta_max
        new_params = self.current_params + deltas

        # Check for illegal states (parameters outside valid ranges)
        illegal = np.any(new_params < self.param_min) or np.any(new_params > self.param_max)

        # Clip to valid range
        self.current_params = np.clip(new_params, self.param_min, self.param_max).astype(np.float32)

        # Compute new observation
        obs = self._compute_observation()
        cd, downforce, efficiency = obs[0], obs[1], obs[2]

        # =====================================================================
        # REWARD FUNCTION v2 — Normalised, multi-component, efficiency-centric
        #
        # Design goals:
        #   - Total reward per step in roughly [-2, +2] range
        #   - Primary signal: how close is efficiency to target (1.5)?
        #   - Secondary signals: beat drag/downforce baselines
        #   - Bonuses: efficiency improvement, new all-time best
        #   - Penalties: parameter oscillation, illegal states
        # =====================================================================
        reward = 0.0

        # --- Component 1: Efficiency score (primary) ---
        # Normalized score: 0 at baseline (~1.0), 1.0 at target
        eff_score = (efficiency - 1.0) / (self.efficiency_target - 1.0)
        eff_score = np.clip(eff_score, -1.0, 2.0)
        reward += self.efficiency_weight * eff_score * 0.1  # scaled to ~[-0.5, +1.0]

        # --- Component 2: Drag below baseline (lower is better) ---
        drag_improvement = (self.drag_baseline - cd) / self.drag_baseline  # positive when Cd < baseline
        reward += self.drag_penalty * drag_improvement * (-1.0)  # drag_penalty is negative, so double-neg = pos

        # --- Component 3: Downforce above baseline (higher is better) ---
        df_improvement = (downforce - self.downforce_baseline) / self.downforce_baseline
        reward += self.downforce_weight * np.clip(df_improvement, -1.0, 2.0) * 0.1

        # --- Component 4: Efficiency improvement over previous step ---
        eff_delta = efficiency - self.prev_efficiency
        reward += self.efficiency_improvement_bonus * np.clip(eff_delta, -0.5, 0.5)

        # --- Component 5: New all-time best efficiency ---
        if efficiency > self.best_efficiency:
            improvement_magnitude = efficiency - self.best_efficiency
            reward += self.new_best_bonus * (1.0 + improvement_magnitude * 5.0)
            self.best_efficiency = efficiency

        # --- Component 6: Smoothness penalty ---
        # Penalize large changes in action from previous step (prevents oscillation)
        action_change = np.mean(np.abs(action - self.prev_action))
        reward += self.smoothness_penalty * action_change

        # --- Component 7: Illegal state penalty ---
        if illegal:
            reward += self.illegal_state_penalty

        # Update tracking state
        self.prev_efficiency = efficiency
        self.prev_action = action.copy()
        self.current_step += 1

        # Termination conditions
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = self._build_info(obs)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            obs = self._compute_observation()
            lines = [
                "=" * 50,
                f"Step {self.current_step}/{self.max_steps}",
                f"Cd: {obs[0]:.4f}  Downforce(-Cl): {obs[1]:.4f}  Efficiency: {obs[2]:.4f}",
                "Parameters:",
            ]
            for i, name in enumerate(self.param_names):
                lines.append(f"  {name}: {self.current_params[i]:.2f}")
            lines.append("=" * 50)
            output = "\n".join(lines)
            print(output)
            return output
        return None

    # ------------------------------------------------------------------
    # Surrogate CFD Model
    # ------------------------------------------------------------------

    def _compute_observation(self) -> np.ndarray:
        """
        Analytical surrogate CFD model.

        Approximates aerodynamic coefficients from geometry parameters using
        physically-motivated empirical relationships. This is intentionally
        simplified but captures the key trade-offs:
            - Higher wing angles → more downforce but more drag
            - Lower ride height → more ground effect downforce
            - Larger diffuser → more rear downforce
            - Gurney flaps → sharp downforce gain with drag cost
        """
        p = self.current_params
        # Unpack for readability
        fwa = p[0]  # front_wing_angle
        rwa = p[1]  # rear_wing_angle
        da = p[2]   # diffuser_angle
        rh = p[3]   # ride_height
        nh = p[4]   # nose_height
        sps = p[5]  # side_pod_shape
        feh = p[6]  # floor_edge_height
        gfs = p[7]  # gurney_flap_size

        # --- Drag Coefficient (Cd) ---
        # Base drag
        cd_base = 0.30

        # Wing drag contribution (roughly proportional to sin^2 of angle)
        cd_front_wing = 0.012 * np.sin(np.radians(fwa)) ** 2
        cd_rear_wing = 0.018 * np.sin(np.radians(rwa)) ** 2

        # Diffuser-induced drag
        cd_diffuser = 0.004 * (da / 15.0)

        # Ride height effects (lower → less frontal area drag but more ground friction)
        cd_ride = 0.002 * (1.0 - (rh - 50.0) / 70.0)

        # Nose shape contribution
        cd_nose = 0.003 * (nh / 250.0)

        # Side pod shape (streamlined shapes reduce drag)
        cd_sidepod = 0.005 * (1.0 - sps / 5.0)

        # Floor edge vortex drag
        cd_floor = 0.002 * (feh / 50.0)

        # Gurney flap drag
        cd_gurney = 0.008 * (gfs / 15.0) ** 1.5

        cd = (cd_base + cd_front_wing + cd_rear_wing + cd_diffuser +
              cd_ride + cd_nose + cd_sidepod + cd_floor + cd_gurney)

        # Add small noise for realism
        cd += self.np_random.normal(0, 0.001)
        cd = max(cd, 0.05)  # Physical floor

        # --- Downforce Coefficient (-Cl, positive means downforce) ---
        # Front wing downforce
        cl_front_wing = 0.15 * np.sin(np.radians(fwa * 2.0))

        # Rear wing downforce (dominant contributor)
        cl_rear_wing = 0.25 * np.sin(np.radians(rwa * 1.5))

        # Ground effect (lower ride height → exponentially more downforce)
        ground_effect = 0.3 * np.exp(-rh / 80.0)

        # Diffuser downforce
        cl_diffuser = 0.12 * np.sin(np.radians(da * 3.0))

        # Gurney flap (effective at generating downforce)
        cl_gurney = 0.06 * (gfs / 15.0)

        # Floor edge vortices enhance diffuser
        cl_floor = 0.04 * (1.0 - feh / 50.0)

        # Side pod doesn't contribute much to downforce
        cl_sidepod = 0.01 * (sps / 5.0)

        downforce = (cl_front_wing + cl_rear_wing + ground_effect +
                     cl_diffuser + cl_gurney + cl_floor + cl_sidepod)

        # Add small noise
        downforce += self.np_random.normal(0, 0.002)
        downforce = max(downforce, 0.0)

        # --- Derived Metrics ---
        efficiency = downforce / cd if cd > 0.001 else 0.0

        # Pressure recovery (higher with better diffuser design)
        pressure_recovery = 0.6 + 0.3 * np.sin(np.radians(da * 4.0)) - 0.1 * (rh / 120.0)
        pressure_recovery = np.clip(pressure_recovery, 0.0, 1.0)

        # Flow velocity ratio (how well flow is accelerated under floor)
        flow_velocity_ratio = 1.2 + 0.5 * np.exp(-rh / 60.0) - 0.1 * (feh / 50.0)

        # Flow separation index (lower is better; high wing angles cause separation)
        separation = 0.1 + 0.3 * (fwa / 15.0) ** 2 + 0.4 * (rwa / 25.0) ** 2
        separation = np.clip(separation, 0.0, 1.0)

        # Turbulence intensity
        turbulence = 0.05 + 0.1 * (rwa / 25.0) + 0.05 * (gfs / 15.0)

        # Surface friction coefficient
        surface_friction = 0.003 + 0.001 * (1.0 - sps / 5.0) + 0.002 * (rh / 120.0)

        obs = np.array([
            cd,
            downforce,
            efficiency,
            pressure_recovery,
            flow_velocity_ratio,
            separation,
            turbulence,
            surface_friction,
        ], dtype=np.float32)

        return obs

    def _build_info(self, obs: np.ndarray) -> Dict[str, Any]:
        """Build the info dict returned by step/reset."""
        param_dict = {name: float(self.current_params[i])
                      for i, name in enumerate(self.param_names)}
        return {
            "parameters": param_dict,
            "cd": float(obs[0]),
            "downforce": float(obs[1]),
            "efficiency": float(obs[2]),
            "best_efficiency": float(self.best_efficiency),
            "step": self.current_step,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_current_parameters(self) -> Dict[str, float]:
        """Return current geometry parameters as a dict."""
        return {name: float(self.current_params[i])
                for i, name in enumerate(self.param_names)}

    def set_parameters(self, params: Dict[str, float]) -> np.ndarray:
        """Set parameters directly and return the observation."""
        for i, name in enumerate(self.param_names):
            if name in params:
                self.current_params[i] = np.clip(
                    params[name], self.param_min[i], self.param_max[i]
                )
        return self._compute_observation()
