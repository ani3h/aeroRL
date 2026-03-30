"""
AeroRL — Visual Wind Tunnel Dashboard

Flask application providing an interactive web interface for:
  - Surrogate CFD wind tunnel simulation
  - Training results visualization
  - Parameter comparison (RL-optimized vs baseline vs custom)
"""

import os
import sys
import glob
import json
import yaml
import logging
from typing import Optional
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from webapp.surrogate import (
    compute_aero,
    parameter_sweep,
    PARAM_DEFAULTS,
    PARAM_RANGES,
    PARAM_NAMES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load project config.yaml."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _find_latest_run() -> Optional[str]:
    """Find the most recent training run directory."""
    logs_dir = os.path.join(PROJECT_ROOT, "results", "logs")
    if not os.path.isdir(logs_dir):
        return None
    runs = sorted(glob.glob(os.path.join(logs_dir, "*")), reverse=True)
    return runs[0] if runs else None


def _load_best_params() -> dict:
    """Load best RL parameters from saved JSON or training data."""
    # 1. Try best_params.json (saved after training/evaluation)
    json_path = os.path.join(PROJECT_ROOT, "results", "models", "best_params.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            params = data.get("parameters", {})
            if params and all(pname in params for pname in PARAM_NAMES):
                logger.info(f"Loaded RL params from {json_path} (eff={data.get('efficiency', '?')})")
                return params
        except Exception:
            pass

    # 2. Try episode_data.csv
    run_dir = _find_latest_run()
    if run_dir:
        csv_path = os.path.join(run_dir, "episode_data.csv")
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                if "efficiency" in df.columns:
                    best_row = df.loc[df["efficiency"].idxmax()]
                    params = {}
                    for pname in PARAM_NAMES:
                        if pname in best_row:
                            params[pname] = float(best_row[pname])
                    if params:
                        return params
            except Exception:
                pass

    # 3. Fall back to defaults
    return PARAM_DEFAULTS.copy()


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Home dashboard page."""
    config = _load_config()
    run_dir = _find_latest_run()

    # Gather training stats
    training_info = {
        "run_id": os.path.basename(run_dir) if run_dir else "No training data",
        "has_data": run_dir is not None,
    }

    # Load metrics CSV if available
    if run_dir:
        csv_path = os.path.join(run_dir, "training_metrics.csv")
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                training_info["total_episodes"] = len(df)
                if "loss" in df.columns:
                    training_info["final_loss"] = f"{df['loss'].dropna().iloc[-1]:.2f}"
            except Exception:
                pass

        ep_csv = os.path.join(run_dir, "episode_data.csv")
        if os.path.exists(ep_csv):
            try:
                import pandas as pd
                df = pd.read_csv(ep_csv)
                if "efficiency" in df.columns:
                    training_info["best_efficiency"] = f"{df['efficiency'].max():.4f}"
                    training_info["total_episodes"] = len(df)
                if "total_reward" in df.columns:
                    training_info["mean_reward"] = f"{df['total_reward'].mean():.2f}"
            except Exception:
                pass

    # Compute baseline aero
    baseline_aero = compute_aero(PARAM_DEFAULTS)

    # RL-optimized params (best from training or defaults)
    rl_params = _load_best_params()
    rl_aero = compute_aero(rl_params)

    return render_template(
        "dashboard.html",
        training_info=training_info,
        baseline_aero=baseline_aero,
        rl_aero=rl_aero,
        rl_params=rl_params,
        config=config,
    )


@app.route("/wind-tunnel")
def wind_tunnel():
    """Interactive wind tunnel page."""
    rl_params = _load_best_params()
    return render_template(
        "wind_tunnel.html",
        param_defaults=PARAM_DEFAULTS,
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        rl_params=rl_params,
    )


@app.route("/compare")
def compare():
    """Comparison view: baseline vs RL vs custom."""
    rl_params = _load_best_params()
    return render_template(
        "compare.html",
        baseline_params=PARAM_DEFAULTS,
        rl_params=rl_params,
        param_names=PARAM_NAMES,
        param_ranges=PARAM_RANGES,
    )


@app.route("/training")
def training():
    """Training curves viewer."""
    run_dir = _find_latest_run()
    plot_files = []
    run_id = None

    if run_dir:
        run_id = os.path.basename(run_dir)
        plots_dir = os.path.join(run_dir, "plots")
        if os.path.isdir(plots_dir):
            for fname in sorted(os.listdir(plots_dir)):
                if fname.endswith(".png"):
                    plot_files.append(fname)

    return render_template(
        "training.html",
        plot_files=plot_files,
        run_id=run_id,
    )


@app.route("/training/plots/<filename>")
def serve_plot(filename):
    """Serve training plot images."""
    run_dir = _find_latest_run()
    if run_dir:
        plots_dir = os.path.join(run_dir, "plots")
        return send_from_directory(plots_dir, filename)
    return "Not found", 404


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Run surrogate CFD simulation with given parameters."""
    data = request.get_json() or {}
    params = {}
    for pname in PARAM_NAMES:
        if pname in data:
            pmin, pmax = PARAM_RANGES[pname]
            params[pname] = max(pmin, min(pmax, float(data[pname])))
        else:
            params[pname] = PARAM_DEFAULTS[pname]

    result = compute_aero(params)
    result["parameters"] = params
    return jsonify(result)


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    """Run parameter sweep for a given parameter."""
    data = request.get_json() or {}
    sweep_param = data.get("param", "rear_wing_angle")
    if sweep_param not in PARAM_NAMES:
        return jsonify({"error": f"Unknown parameter: {sweep_param}"}), 400

    base_params = {}
    for pname in PARAM_NAMES:
        if pname in data:
            base_params[pname] = float(data[pname])
        else:
            base_params[pname] = PARAM_DEFAULTS[pname]

    result = parameter_sweep(base_params, sweep_param, n_points=50)
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Compare baseline, RL, and custom parameters."""
    data = request.get_json() or {}

    baseline = compute_aero(PARAM_DEFAULTS)
    baseline["label"] = "Baseline"

    rl_params = _load_best_params()
    rl = compute_aero(rl_params)
    rl["label"] = "RL-Optimized"

    custom_params = {}
    for pname in PARAM_NAMES:
        if pname in data:
            custom_params[pname] = float(data[pname])
        else:
            custom_params[pname] = PARAM_DEFAULTS[pname]
    custom = compute_aero(custom_params)
    custom["label"] = "Custom"

    return jsonify({
        "baseline": baseline,
        "rl": rl,
        "custom": custom,
        "rl_params": rl_params,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dashboard(host: str = "0.0.0.0", port: int = 5000, debug: bool = True):
    """Launch the Flask dashboard."""
    print(f"\n{'='*60}")
    print(f"  AeroRL Wind Tunnel Dashboard")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard()
