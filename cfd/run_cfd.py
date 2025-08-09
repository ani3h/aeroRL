"""
CFD Workflow Runner

This script demonstrates the complete CFD simulation workflow by integrating:
1. Geometry generation based on F1 car parameters
2. Running CFD simulations using OpenFOAM or SimScale
3. Parsing and analyzing results for aerodynamic performance metrics
"""

import os
import argparse
import logging
import json
import time
from typing import Dict, List, Any

# Import our CFD modules
from geometry_generator import F1GeometryGenerator
from run_simulation import CFDSimulator
from parse_results import CFDResultsParser

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_directories() -> Dict[str, str]:
    # Define base directories
    base_dir = os.path.dirname(os.path.abspath(__file__))

    directories = {
        'base': base_dir,
        'models': os.path.join(base_dir, 'models'),
        'geometries': os.path.join(base_dir, 'geometries'),
        'base_case': os.path.join(base_dir, 'base_case'),
        'simulations': os.path.join(base_dir, 'simulations'),
        'results': os.path.join(base_dir, 'results')
    }

    # Create directories if they don't exist
    for dir_path in directories.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

    return directories


def run_single_simulation(params: Dict[str, float], dirs: Dict[str, str], sim_type: str) -> Dict[str, Any]:
    # Step 1: Generate geometry
    logger.info("Generating F1 car geometry...")
    generator = F1GeometryGenerator(
        base_model_path=os.path.join(dirs['models'], 'base_f1_car.stl'),
        output_dir=dirs['geometries']
    )
    geometry_file = generator.generate_geometry(params)
    logger.info(f"Generated geometry: {geometry_file}")

    # Step 2: Run CFD simulation
    logger.info(f"Running {sim_type} simulation...")
    simulator = CFDSimulator(
        sim_type=sim_type,
        base_case_dir=dirs['base_case'],
        output_dir=dirs['simulations']
    )

    # Prepare the case
    case_dir = simulator.prepare_case(geometry_file, params)
    logger.info(f"Prepared case directory: {case_dir}")

    # Run the simulation
    sim_result = simulator.run_simulation(case_dir)
    if sim_result['success']:
        logger.info(
            f"Simulation completed successfully in {sim_result['duration']:.2f} seconds")
    else:
        logger.error(
            f"Simulation failed: {sim_result.get('error', 'Unknown error')}")
        return {'success': False, 'error': sim_result.get('error', 'Unknown error')}

    # Step 3: Parse results
    logger.info("Parsing simulation results...")
    parser = CFDResultsParser(
        results_dir=dirs['simulations'],
        output_dir=dirs['results']
    )

    parsed_result = parser.parse_case(case_dir)

    # Generate plots
    plot_path = parser.plot_results(parsed_result)
    logger.info(f"Generated result plot: {plot_path}")

    # Combine all results
    final_result = {
        'success': True,
        'geometry_file': geometry_file,
        'case_dir': case_dir,
        'parameters': params,
        'aero_metrics': parsed_result['results'],
        'plot_path': plot_path
    }

    # Save final result to JSON
    result_file = os.path.join(
        dirs['results'], f"sim_result_{int(time.time())}.json")
    with open(result_file, 'w') as f:
        json.dump(final_result, f, indent=4)

    return final_result


def parameter_sweep(base_params: Dict[str, float], sweep_param: str,
                    values: List[float], dirs: Dict[str, str], sim_type: str) -> List[Dict[str, Any]]:
    results = []

    for value in values:
        # Create a copy of base parameters and update the sweep parameter
        params = base_params.copy()
        params[sweep_param] = value

        logger.info(f"Running simulation with {sweep_param} = {value}")
        result = run_single_simulation(params, dirs, sim_type)

        if result['success']:
            results.append(result)
        else:
            logger.error(
                f"Skipping failed simulation with {sweep_param} = {value}")

    # If we have results, generate comparison plots
    if results:
        logger.info("Generating comparison plots...")
        parser = CFDResultsParser(
            results_dir=dirs['simulations'],
            output_dir=dirs['results']
        )
        parser._create_parameter_sensitivity_plot(results, sweep_param)

    return results


def main():
    parser = argparse.ArgumentParser(description='CFD Workflow Runner')

    parser.add_argument('--sim-type', type=str, default='openfoam_local',
                        choices=['openfoam_local',
                                 'openfoam_docker', 'simscale_api'],
                        help='Type of simulation to run')

    parser.add_argument('--sweep', action='store_true',
                        help='Run a parameter sweep instead of a single simulation')

    parser.add_argument('--param', type=str, default='rear_wing_angle',
                        help='Parameter to sweep (only used with --sweep)')

    parser.add_argument('--min', type=float, default=5,
                        help='Minimum value for parameter sweep')

    parser.add_argument('--max', type=float, default=25,
                        help='Maximum value for parameter sweep')

    parser.add_argument('--steps', type=int, default=5,
                        help='Number of steps in parameter sweep')

    args = parser.parse_args()

    # Set up directories
    dirs = setup_directories()

    # Define default parameters for F1 car
    default_params = {
        'front_wing_angle': 10.0,    # degrees
        'rear_wing_angle': 15.0,     # degrees
        'diffuser_angle': 10.0,      # degrees
        'ride_height': 80.0,         # mm
        'nose_height': 150.0,        # mm
        'side_pod_shape': 3.0,       # shape index
        'floor_edge_height': 25.0,   # mm
        'gurney_flap_size': 5.0      # mm
    }

    # Run simulation(s)
    if args.sweep:
        # Calculate parameter values for sweep
        values = [args.min + i * (args.max - args.min) /
                  (args.steps - 1) for i in range(args.steps)]

        logger.info(
            f"Running parameter sweep on {args.param} with values: {values}")
        results = parameter_sweep(
            default_params, args.param, values, dirs, args.sim_type)

        # Print summary
        logger.info(f"Completed {len(results)} simulations")

        if results:
            # Print best result
            best_result = max(
                results, key=lambda r: r['aero_metrics'].get('efficiency', 0))
            best_value = best_result['parameters'][args.param]
            best_efficiency = best_result['aero_metrics'].get('efficiency', 0)

            logger.info(
                f"Best result: {args.param} = {best_value} with efficiency = {best_efficiency:.4f}")

    else:
        # Run a single simulation with default parameters
        logger.info("Running single simulation with default parameters")
        result = run_single_simulation(default_params, dirs, args.sim_type)

        if result['success']:
            # Print results
            cd = result['aero_metrics'].get('cd')
            cl = result['aero_metrics'].get('cl')
            efficiency = result['aero_metrics'].get('efficiency')

            logger.info("Simulation Results:")
            logger.info(f"  Drag Coefficient (Cd): {cd:.4f}")
            logger.info(f"  Lift Coefficient (Cl): {cl:.4f}")
            logger.info(f"  Efficiency (Downforce/Drag): {efficiency:.4f}")
            logger.info(f"  Plot: {result['plot_path']}")


if __name__ == "__main__":
    main()
