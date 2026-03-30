"""
Standalone Surrogate CFD Model

Extracted from environment/f1_env.py so the Flask dashboard can compute
aerodynamic coefficients without importing the full Gymnasium environment.
"""

import numpy as np
from typing import Dict, Tuple

# Parameter definitions: (name, min, max, default)
PARAM_DEFS = [
    ("front_wing_angle", -5.0, 15.0, 10.0),
    ("rear_wing_angle", 0.0, 25.0, 15.0),
    ("diffuser_angle", 3.0, 18.0, 10.0),
    ("ride_height", 50.0, 120.0, 80.0),
    ("nose_height", 100.0, 250.0, 150.0),
    ("side_pod_shape", 1.0, 5.0, 3.0),
    ("floor_edge_height", 10.0, 50.0, 25.0),
    ("gurney_flap_size", 0.0, 15.0, 5.0),
]

PARAM_NAMES = [p[0] for p in PARAM_DEFS]
PARAM_DEFAULTS = {p[0]: p[3] for p in PARAM_DEFS}
PARAM_RANGES = {p[0]: (p[1], p[2]) for p in PARAM_DEFS}


def compute_aero(params: Dict[str, float], add_noise: bool = False) -> Dict[str, float]:
    """
    Compute aerodynamic coefficients from geometry parameters.

    Args:
        params: dict mapping parameter names to values
        add_noise: if True, add small Gaussian noise for realism

    Returns:
        dict with cd, downforce, efficiency, pressure_recovery,
        flow_velocity_ratio, separation, turbulence, surface_friction
    """
    fwa = params.get("front_wing_angle", 10.0)
    rwa = params.get("rear_wing_angle", 15.0)
    da = params.get("diffuser_angle", 10.0)
    rh = params.get("ride_height", 80.0)
    nh = params.get("nose_height", 150.0)
    sps = params.get("side_pod_shape", 3.0)
    feh = params.get("floor_edge_height", 25.0)
    gfs = params.get("gurney_flap_size", 5.0)

    # --- Drag Coefficient (Cd) ---
    cd_base = 0.30
    cd_front_wing = 0.012 * np.sin(np.radians(fwa)) ** 2
    cd_rear_wing = 0.018 * np.sin(np.radians(rwa)) ** 2
    cd_diffuser = 0.004 * (da / 15.0)
    cd_ride = 0.002 * (1.0 - (rh - 50.0) / 70.0)
    cd_nose = 0.003 * (nh / 250.0)
    cd_sidepod = 0.005 * (1.0 - sps / 5.0)
    cd_floor = 0.002 * (feh / 50.0)
    cd_gurney = 0.008 * (gfs / 15.0) ** 1.5

    cd = (cd_base + cd_front_wing + cd_rear_wing + cd_diffuser +
          cd_ride + cd_nose + cd_sidepod + cd_floor + cd_gurney)

    if add_noise:
        cd += np.random.normal(0, 0.001)
    cd = max(cd, 0.05)

    # --- Downforce Coefficient (-Cl) ---
    cl_front_wing = 0.15 * np.sin(np.radians(fwa * 2.0))
    cl_rear_wing = 0.25 * np.sin(np.radians(rwa * 1.5))
    ground_effect = 0.3 * np.exp(-rh / 80.0)
    cl_diffuser = 0.12 * np.sin(np.radians(da * 3.0))
    cl_gurney = 0.06 * (gfs / 15.0)
    cl_floor = 0.04 * (1.0 - feh / 50.0)
    cl_sidepod = 0.01 * (sps / 5.0)

    downforce = (cl_front_wing + cl_rear_wing + ground_effect +
                 cl_diffuser + cl_gurney + cl_floor + cl_sidepod)

    if add_noise:
        downforce += np.random.normal(0, 0.002)
    downforce = max(downforce, 0.0)

    # --- Derived Metrics ---
    efficiency = downforce / cd if cd > 0.001 else 0.0

    pressure_recovery = 0.6 + 0.3 * np.sin(np.radians(da * 4.0)) - 0.1 * (rh / 120.0)
    pressure_recovery = float(np.clip(pressure_recovery, 0.0, 1.0))

    flow_velocity_ratio = 1.2 + 0.5 * np.exp(-rh / 60.0) - 0.1 * (feh / 50.0)

    separation = 0.1 + 0.3 * (fwa / 15.0) ** 2 + 0.4 * (rwa / 25.0) ** 2
    separation = float(np.clip(separation, 0.0, 1.0))

    turbulence = 0.05 + 0.1 * (rwa / 25.0) + 0.05 * (gfs / 15.0)

    surface_friction = 0.003 + 0.001 * (1.0 - sps / 5.0) + 0.002 * (rh / 120.0)

    # Drag breakdown for visualization
    drag_breakdown = {
        "base": round(cd_base, 5),
        "front_wing": round(cd_front_wing, 5),
        "rear_wing": round(cd_rear_wing, 5),
        "diffuser": round(cd_diffuser, 5),
        "ride_height": round(cd_ride, 5),
        "nose": round(cd_nose, 5),
        "side_pod": round(cd_sidepod, 5),
        "floor_edge": round(cd_floor, 5),
        "gurney_flap": round(cd_gurney, 5),
    }

    # Downforce breakdown
    downforce_breakdown = {
        "front_wing": round(cl_front_wing, 5),
        "rear_wing": round(cl_rear_wing, 5),
        "ground_effect": round(float(ground_effect), 5),
        "diffuser": round(cl_diffuser, 5),
        "gurney_flap": round(cl_gurney, 5),
        "floor_edge": round(cl_floor, 5),
        "side_pod": round(cl_sidepod, 5),
    }

    return {
        "cd": round(float(cd), 5),
        "downforce": round(float(downforce), 5),
        "efficiency": round(float(efficiency), 4),
        "pressure_recovery": round(float(pressure_recovery), 4),
        "flow_velocity_ratio": round(float(flow_velocity_ratio), 4),
        "separation": round(float(separation), 4),
        "turbulence": round(float(turbulence), 4),
        "surface_friction": round(float(surface_friction), 5),
        "drag_breakdown": drag_breakdown,
        "downforce_breakdown": downforce_breakdown,
    }


def parameter_sweep(
    base_params: Dict[str, float],
    sweep_param: str,
    n_points: int = 50,
) -> Dict[str, list]:
    """
    Sweep a single parameter across its full range while keeping others fixed.

    Returns dict of lists: param_values, cd, downforce, efficiency.
    """
    pmin, pmax = PARAM_RANGES[sweep_param]
    values = np.linspace(pmin, pmax, n_points).tolist()

    cd_list, df_list, eff_list = [], [], []
    for val in values:
        p = dict(base_params)
        p[sweep_param] = val
        result = compute_aero(p)
        cd_list.append(result["cd"])
        df_list.append(result["downforce"])
        eff_list.append(result["efficiency"])

    return {
        "param_name": sweep_param,
        "values": [round(v, 2) for v in values],
        "cd": cd_list,
        "downforce": df_list,
        "efficiency": eff_list,
    }
