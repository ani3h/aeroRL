"""
CFD Results Parser

This module parses CFD simulation results to extract key aerodynamic metrics like 
drag coefficient (Cd) and downforce coefficient (Cl) from OpenFOAM or SimScale output.
"""

import os
import re
import csv
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CFDResultsParser:

    def __init__(self, results_dir: str, output_dir: Optional[str] = None):
        self.results_dir = results_dir
        self.output_dir = output_dir or os.path.join(results_dir, "processed")

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

        logger.info(f"CFDResultsParser initialized for {results_dir}")

    def parse_case(self, case_dir: str) -> Dict[str, Any]:
        logger.info(f"Parsing results from {case_dir}")

        # Check if this is a valid case directory
        if not os.path.isdir(case_dir):
            logger.error(f"Invalid case directory: {case_dir}")
            return {'error': 'Invalid case directory'}

        # Get case parameters
        params = self._get_case_parameters(case_dir)

        # Determine the simulation type based on directory structure
        if os.path.exists(os.path.join(case_dir, 'postProcessing')):
            # Appears to be an OpenFOAM case
            results = self._parse_openfoam_results(case_dir)
        elif os.path.exists(os.path.join(case_dir, 'simscale_results')):
            # Appears to be a SimScale case
            results = self._parse_simscale_results(case_dir)
        else:
            # Try generic parsing
            results = self._parse_generic_results(case_dir)

        # Combine parameters and results
        output = {
            'case_dir': case_dir,
            'parameters': params,
            'results': results,
            'timestamp': os.path.getmtime(case_dir)
        }

        # Save results to JSON
        results_file = os.path.join(
            self.output_dir, f"{os.path.basename(case_dir)}_results.json")
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=4)

        logger.info(f"Saved parsed results to {results_file}")

        return output

    def _get_case_parameters(self, case_dir: str) -> Dict[str, Any]:
        # Check for a params.json file
        params_file = os.path.join(case_dir, 'params.json')
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                return json.load(f)

        # If no params file exists, try to extract from other files
        params = {}

        # Example: extract from controlDict
        control_dict_path = os.path.join(case_dir, 'system', 'controlDict')
        if os.path.exists(control_dict_path):
            with open(control_dict_path, 'r') as f:
                content = f.read()
                # Extract parameters from comments we added during case preparation
                param_lines = re.findall(r'// ([^:]+): (.+)', content)
                for key, value in param_lines:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value

        return params

    def _parse_openfoam_results(self, case_dir: str) -> Dict[str, Any]:
        results = {
            'converged': False,
            'cd': None,  # Drag coefficient
            'cl': None,  # Lift coefficient (negative for downforce)
            'efficiency': None,  # Downforce to drag ratio
            'pressure_field': None,
            'velocity_field': None,
            'iterations': 0
        }

        # Check for forces directory in postProcessing
        forces_dir = os.path.join(case_dir, 'postProcessing', 'forceCoeffs')
        if os.path.exists(forces_dir):
            # Find the latest time directory
            time_dirs = [d for d in os.listdir(
                forces_dir) if os.path.isdir(os.path.join(forces_dir, d))]
            if time_dirs:
                latest_time = sorted(time_dirs)[-1]
                coeffs_file = os.path.join(
                    forces_dir, latest_time, 'coefficient.dat')

                if os.path.exists(coeffs_file):
                    # Parse the coefficients file
                    with open(coeffs_file, 'r') as f:
                        lines = f.readlines()

                    # Find header and data
                    header_line = None
                    data_lines = []

                    for i, line in enumerate(lines):
                        if line.startswith('#'):
                            header_line = line
                        elif not line.startswith('#') and line.strip():
                            data_lines.append(line)

                    if header_line and data_lines:
                        # Parse header to find column indices
                        headers = header_line.strip('# \n').split()
                        cd_index = None
                        cl_index = None

                        for i, header in enumerate(headers):
                            if header.lower() == 'cd' or header.lower() == 'force(d)':
                                cd_index = i
                            elif header.lower() == 'cl' or header.lower() == 'force(l)':
                                cl_index = i

                        if cd_index is not None and cl_index is not None:
                            # Use last line for final values
                            final_values = data_lines[-1].strip().split()
                            results['cd'] = float(final_values[cd_index])
                            results['cl'] = float(final_values[cl_index])
                            results['iterations'] = len(data_lines)

                            # Calculate efficiency (downforce to drag ratio)
                            # Note: Cl is negative for downforce, so we negate it
                            if results['cd'] != 0:
                                results['efficiency'] = - \
                                    (results['cl'] / results['cd'])

                            results['converged'] = True

        # If we couldn't find force coefficients, try to parse residuals to check convergence
        if not results['converged']:
            residuals_dir = os.path.join(
                case_dir, 'postProcessing', 'residuals')
            if os.path.exists(residuals_dir):
                # Check convergence based on residuals
                # Implementation would depend on specific convergence criteria
                results['converged'] = self._check_convergence_from_residuals(
                    residuals_dir)

        return results

    def _parse_generic_results(self, case_dir: str) -> Dict[str, Any]:
        results = {
            'converged': False,
            'cd': None,
            'cl': None,
            'efficiency': None
        }

        # Look for any file containing 'force' or 'coeff' in the name
        for root, _, files in os.walk(case_dir):
            for filename in files:
                if 'force' in filename.lower() or 'coeff' in filename.lower():
                    file_path = os.path.join(root, filename)

                    # Try to parse as text file
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()

                            # Look for drag coefficient
                            cd_match = re.search(
                                r'(?:Cd|CD|drag coefficient)[:\s=]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', content)
                            if cd_match:
                                results['cd'] = float(cd_match.group(1))

                            # Look for lift coefficient
                            cl_match = re.search(
                                r'(?:Cl|CL|lift coefficient)[:\s=]+([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', content)
                            if cl_match:
                                results['cl'] = float(cl_match.group(1))

                            # If we found both, calculate efficiency
                            if results['cd'] is not None and results['cl'] is not None:
                                if results['cd'] != 0:
                                    results['efficiency'] = - \
                                        (results['cl'] / results['cd'])
                                results['converged'] = True
                                break
                    except:
                        # If we can't parse as text, just continue
                        continue

        return results

    def _check_convergence_from_residuals(self, residuals_dir: str) -> bool:
        # In a real implementation, this would analyze residual trends
        # For now, we'll use a simple implementation

        # Find the latest time directory
        time_dirs = [d for d in os.listdir(residuals_dir) if os.path.isdir(
            os.path.join(residuals_dir, d))]
        if not time_dirs:
            return False

        latest_time = sorted(time_dirs)[-1]
        residuals_file = os.path.join(
            residuals_dir, latest_time, 'residuals.dat')

        if not os.path.exists(residuals_file):
            return False

        # Read the last few residual values
        try:
            with open(residuals_file, 'r') as f:
                lines = f.readlines()

            # Skip header lines
            data_lines = [line for line in lines if not line.startswith('#')]

            if len(data_lines) < 10:
                return False  # Not enough iterations to determine convergence

            # Get the last 10 iterations
            last_iterations = data_lines[-10:]

            # Parse values for key residuals (e.g., Ux, Uy, Uz, p)
            residual_values = []
            for line in last_iterations:
                values = line.strip().split()
                if len(values) >= 2:  # At least iteration number and one residual
                    try:
                        # Assuming the second column is the first residual value
                        residual_values.append(float(values[1]))
                    except:
                        continue

            if not residual_values:
                return False

            # Check if last values are below threshold
            threshold = 1e-4  # Typical convergence criterion
            if max(residual_values) < threshold:
                return True

            # Check if values are stabilized (not changing much)
            if max(residual_values) - min(residual_values) < 0.1 * max(residual_values):
                return True

        except Exception as e:
            logger.error(f"Error checking convergence: {e}")

        return False

    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        if save_path is None:
            save_path = os.path.join(
                self.output_dir, f"aero_efficiency_{os.path.basename(results['case_dir'])}.png")

        # Create the figure
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Get parameters and results
        params = results['parameters']
        cd = results['results'].get('cd')
        cl = results['results'].get('cl')
        efficiency = results['results'].get('efficiency')

        # Title with key parameters
        param_str = ", ".join([f"{k}: {v:.2f}" for k, v in params.items() if k in [
                              'front_wing_angle', 'rear_wing_angle', 'ride_height']])
        plt.title(f"Aerodynamic Efficiency\n{param_str}", fontsize=14)

        # Create bar chart for Cd, Cl, and Efficiency
        labels = [
            'Drag Coefficient (Cd)', 'Downforce Coefficient (-Cl)', 'Efficiency (Downforce/Drag)']
        # Negate Cl for downforce
        values = [cd, -cl if cl else None, efficiency]

        # Filter out None values
        valid_labels = [labels[i]
                        for i in range(len(labels)) if values[i] is not None]
        valid_values = [values[i]
                        for i in range(len(values)) if values[i] is not None]

        if not valid_values:
            logger.warning("No valid values to plot")
            return None

        # Create bar chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax1.bar(valid_labels, valid_values,
                       color=colors[:len(valid_values)])

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                     f'{height:.3f}', ha='center', fontsize=10)

        # Customize the plot
        ax1.set_xlabel('Metric', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved plot to {save_path}")
        return save_path

    def batch_process(self) -> List[Dict[str, Any]]:
        all_results = []

        # Look for case directories
        for item in os.listdir(self.results_dir):
            case_dir = os.path.join(self.results_dir, item)
            if os.path.isdir(case_dir):
                try:
                    result = self.parse_case(case_dir)
                    all_results.append(result)

                    # Create a plot for this case
                    self.plot_results(result)

                except Exception as e:
                    logger.error(f"Error processing case {case_dir}: {e}")

        # Sort results by timestamp
        all_results.sort(key=lambda x: x['timestamp'])

        # Save summary to CSV
        self._save_summary_csv(all_results)

        # Create comparison plots
        self._create_comparison_plots(all_results)

        return all_results

    def _save_summary_csv(self, all_results: List[Dict[str, Any]]) -> None:
        csv_path = os.path.join(self.output_dir, "summary_results.csv")

        # Collect all parameter keys
        param_keys = set()
        for result in all_results:
            param_keys.update(result['parameters'].keys())

        # Sort the parameter keys
        param_keys = sorted(list(param_keys))

        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            header = ['Case'] + param_keys + \
                ['Cd', 'Cl', 'Efficiency', 'Converged']
            writer.writerow(header)

            # Write data rows
            for result in all_results:
                case_name = os.path.basename(result['case_dir'])

                # Extract parameter values
                param_values = [result['parameters'].get(
                    key, '') for key in param_keys]

                # Extract result values
                cd = result['results'].get('cd', '')
                cl = result['results'].get('cl', '')
                efficiency = result['results'].get('efficiency', '')
                converged = result['results'].get('converged', False)

                # Write row
                row = [case_name] + param_values + \
                    [cd, cl, efficiency, converged]
                writer.writerow(row)

        logger.info(f"Saved summary CSV to {csv_path}")

    def _create_comparison_plots(self, all_results: List[Dict[str, Any]]) -> None:
        # Only create plots if we have results with valid data
        valid_results = [r for r in all_results if r['results'].get(
            'cd') is not None and r['results'].get('cl') is not None]

        if not valid_results:
            logger.warning("No valid results for comparison plots")
            return

        # Create Cd vs Cl scatter plot
        self._create_cd_cl_plot(valid_results)

        # Create parameter sensitivity plots
        parameter_keys = ['front_wing_angle',
                          'rear_wing_angle', 'ride_height', 'diffuser_angle']
        for param in parameter_keys:
            self._create_parameter_sensitivity_plot(valid_results, param)

    def _create_cd_cl_plot(self, results: List[Dict[str, Any]]) -> None:
        plt.figure(figsize=(10, 8))

        # Extract Cd and Cl values
        cd_values = [r['results']['cd'] for r in results]
        cl_values = [r['results']['cl'] for r in results]
        efficiencies = [r['results'].get('efficiency', 0) for r in results]

        # Create scatter plot
        scatter = plt.scatter(cd_values, cl_values, c=efficiencies, cmap='viridis',
                              s=100, alpha=0.7)

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Efficiency (Downforce/Drag)', fontsize=12)

        # Add case labels
        for i, result in enumerate(results):
            case_name = os.path.basename(result['case_dir'])
            plt.annotate(case_name, (cd_values[i], cl_values[i]),
                         fontsize=8, alpha=0.7)

        # Add plot details
        plt.xlabel('Drag Coefficient (Cd)', fontsize=12)
        plt.ylabel('Lift Coefficient (Cl)', fontsize=12)
        plt.title('Drag vs. Lift Coefficient', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the plot
        plot_path = os.path.join(self.output_dir, "cd_cl_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved Cd vs Cl plot to {plot_path}")

    def _create_parameter_sensitivity_plot(self, results: List[Dict[str, Any]], parameter: str) -> None:
        # Check if we have this parameter in results
        param_values = [r['parameters'].get(parameter) for r in results]
        if not any(param_values):
            logger.info(
                f"Parameter {parameter} not found in results, skipping sensitivity plot")
            return

        # Filter results that have this parameter
        filtered_results = [r for r in results if parameter in r['parameters']]

        # Sort by parameter value
        filtered_results.sort(key=lambda x: x['parameters'][parameter])

        # Extract values
        param_values = [r['parameters'][parameter] for r in filtered_results]
        cd_values = [r['results']['cd'] for r in filtered_results]
        cl_values = [r['results']['cl'] for r in filtered_results]
        eff_values = [r['results'].get('efficiency', 0)
                      for r in filtered_results]

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plot Cd and Cl on left axis
        ax1.set_xlabel(f'{parameter.replace("_", " ").title()}', fontsize=12)
        ax1.set_ylabel('Coefficient Value', fontsize=12)

        line1 = ax1.plot(param_values, cd_values, 'o-',
                         color='blue', label='Cd')
        line2 = ax1.plot(param_values, cl_values,
                         's-', color='red', label='Cl')

        # Create a second y-axis for efficiency
        ax2 = ax1.twinx()
        ax2.set_ylabel('Efficiency (Downforce/Drag)', fontsize=12)
        line3 = ax2.plot(param_values, eff_values, '^-',
                         color='green', label='Efficiency')

        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')

        # Add grid and title
        ax1.grid(True, linestyle='--', alpha=0.7)
        plt.title(
            f'Sensitivity to {parameter.replace("_", " ").title()}', fontsize=14)

        # Save the plot
        plot_path = os.path.join(
            self.output_dir, f"{parameter}_sensitivity.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved {parameter} sensitivity plot to {plot_path}")


# Example usage
if __name__ == "__main__":
    # For testing purposes
    parser = CFDResultsParser(
        results_dir="simulations/results",
        output_dir="results/aero_analysis"
    )

    # Parse a single case
    case_dir = "simulations/results/case_example"
    result = parser.parse_case(case_dir)

    print(f"Drag coefficient (Cd): {result['results'].get('cd')}")
    print(f"Lift coefficient (Cl): {result['results'].get('cl')}")
    print(f"Efficiency: {result['results'].get('efficiency')}")

    # Create a plot
    plot_path = parser.plot_results(result)
    print(f"Plot saved to: {plot_path}")

    # Batch process all cases
    all_results = parser.batch_process()
    print(f"Processed {len(all_results)} cases")

    def _parse_simscale_results(self, case_dir: str) -> Dict[str, Any]:
        results = {
            'converged': False,
            'cd': None,
            'cl': None,
            'efficiency': None
        }

        # SimScale results may be in a different format
        # This implementation would depend on the specific SimScale output format
        simscale_results_dir = os.path.join(case_dir, 'simscale_results')

        # Look for a results CSV file
        csv_files = [f for f in os.listdir(
            simscale_results_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            if 'force' in csv_file.lower() or 'coeff' in csv_file.lower():
                file_path = os.path.join(simscale_results_dir, csv_file)

                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    headers = next(reader)

                    # Find relevant columns
                    cd_index = None
                    cl_index = None

                    for i, header in enumerate(headers):
                        if 'drag' in header.lower() or 'cd' in header.lower():
                            cd_index = i
                        elif 'lift' in header.lower() or 'cl' in header.lower():
                            cl_index = i

                    if cd_index is not None and cl_index is not None:
                        # Read all rows
                        rows = list(reader)
                        if rows:
                            # Use last row for final values
                            final_row = rows[-1]
                            results['cd'] = float(final_row[cd_index])
                            results['cl'] = float(final_row[cl_index])

                            # Calculate efficiency
                            if results['cd'] != 0:
                                results['efficiency'] = - \
                                    (results['cl'] / results['cd'])

                            results['converged'] = True
                            break

        return results
