# 2D Electrostatic Potential with Physics-Informed Neural Networks

This project solves the 2D electrostatic potential problem with multiple point charges using Physics-Informed Neural Networks (PINNs). The implementation is based on DeepXDE, a library specifically designed for PINNs.

## Problem Description

We solve Poisson's equation for the electrostatic potential:

∇²φ(x,y) = -ρ(x,y)/ε₀

Where:
- φ(x,y) is the electrostatic potential
- ρ(x,y) is the charge density
- ε₀ is the permittivity of free space

For point charges, ρ(x,y) is represented as:

ρ(x,y) = q₁δ(x-x₁, y-y₁) + q₂δ(x-x₂, y-y₂) + ...

Where δ is the Dirac delta function.

## Project Structure

```
├── electrostatics-pinn.py           # Main PINN solver
├── electrostatics-config.py         # Configurable PINN solver
├── architecture-comparison.py       # Script to compare PINN architectures
├── poisson-singular-sources.py      # Implementation handling singular sources
└── README.md                        # This file
```

## Installation Requirements

```bash
pip install deepxde tensorflow numpy matplotlib pandas seaborn
```

## Running the Code

### Basic PINN Solver

```bash
python electrostatics-pinn.py
```

This script solves the electrostatic potential problem for two point charges:
- A positive charge at (-0.5, 0)
- A negative charge at (0.5, 0)

The script produces visualizations of:
- The PINN solution
- The analytical solution
- Error plots
- Electric field lines

### Configurable PINN Solver

```bash
python electrostatics-config.py --domain_size 2.0 --num_domain 2000 --num_boundary 200 --epochs 20000 --hidden_layers 4 --neurons 50 --output_dir results
```

This script allows you to customize the PINN architecture, domain size, and other parameters. You can easily modify the charge configuration in the script.

### Architecture Comparison

```bash
python architecture-comparison.py
```

This script compares different PINN architectures (FNN, ResNet, MsFFN) and activation functions (tanh, relu, sin, swish) for solving the electrostatic potential problem. It produces:
- Comparative error metrics
- Performance visualizations
- Bar charts and heatmaps for easy comparison

### Poisson Equation with Singular Sources

```bash
python poisson-singular-sources.py
```

This script demonstrates a more advanced implementation that directly handles the singular sources in Poisson's equation using a smoothed approximation of the Dirac delta function.

## Mathematical Background

### Analytical Solution

For point charges in 2D, the analytical solution for the potential is:

φₐₙₐₗᵧₜᵢ𝒸(x,y) = ∑ᵢ (qᵢ/(2πε₀)) ln(1/((x-xᵢ)² + (y-yᵢ)²))

This solution is used to validate the PINN results.

### Electric Field

The electric field E is calculated as the negative gradient of the potential:

E = -∇φ

## Visualization Results

The code produces several visualizations:

1. **Potential field contours**: Shows the distribution of electric potential in the domain.
2. **3D surface plots**: Visualizes the potential as a 3D surface.
3. **Electric field lines**: Shows the direction and magnitude of the electric field.
4. **Error analysis**: Compares the PINN solution with the analytical solution.

## Extending the Project

You can extend this project in several ways:

1. Add more charges with different configurations
2. Implement different boundary conditions (Neumann, mixed)
3. Extend to 3D electrostatics
4. Include dielectric materials with varying permittivity
5. Implement time-dependent electromagnetics

## Technical Notes

- The implementation uses a neural network with hyperbolic tangent activation functions by default.
- For improved accuracy around singularities, additional points are sampled near charge locations.
- We use both Adam optimizer and L-BFGS for training.
- Electric field computation uses finite differences for visualization purposes.