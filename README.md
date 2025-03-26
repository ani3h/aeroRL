# aeroRL: AI-Powered F1 Aerodynamics Optimization
AeroRL is an AI-driven F1 aerodynamics optimization system that leverages Reinforcement Learning (RL) and Computational Fluid Dynamics (CFD) to minimize drag and maximize downforce in a virtual wind tunnel. This project integrates OpenAI Gym, TensorFlow, and CFD software to iteratively enhance car aerodynamics using AI.

## Features
- AI-Optimized Aerodynamics: Uses RL to adjust F1 car parameters for optimal airflow
- Virtual Wind Tunnel: Runs CFD simulations to evaluate aerodynamic efficiency
- Automated Optimization: AI continuously improves designs based on real-time feedback.
- Python Based: Built using TensorFlow, OpenAI Gym, and CFD tools

## Project Directory Structure
```
AeroRL/
│── models/                # Reinforcement Learning models
│── simulations/           # CFD simulation scripts
│── environment/           # OpenAI Gym environment for aerodynamic optimization
│── training/              # RL training scripts
│── results/               # Logs, graphs, and performance metrics
│── utils/                 # Helper functions and scripts
│── README.md              # Project documentation
│── requirements.txt       # Dependencies
```

## Getting Started

To install the package, you can use pip with the URL of the GitHub repository.

To use the package, you can follow the steps below:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ani3h/aeroRL.git
   cd aeroRL
   ```
   
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Virutal Wind Tunnel Simulation:**
   ```bash
   python simulations/run_cfd.py
   ```

5. **Train the RL Model:**
   ```bash
   python training/train_rl.py
   ```

6. **Test the Optimized Model:**
   ```bash
   python training/test_rl.py
   ```

## Results & Visualization
After training, the system generates:
- Aerodynamic efficiency graphs
- Drag vs downforce optimizations
- Before & after comparisons of CFD results

To Visualize, use:
```bash
python results/plot_results.py
```


