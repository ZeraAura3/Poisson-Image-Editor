import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import time
import os
import pandas as pd
import seaborn as sns

# Enable TensorFlow compatibility mode
tf.compat.v1.disable_eager_execution()

# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

def run_experiment(architecture, activation_func, num_epochs, lr=0.001):
    """
    Run a PINN experiment with the given architecture and activation function
    """
    # Problem parameters
    epsilon_0 = 1.0  # Permittivity of free space (scaled for simplicity)
    domain_size = 2.0  # Domain size [-1, 1] x [-1, 1]
    
    # Define the charges (position and magnitude)
    charges = [
        {"position": (-0.5, 0.0), "magnitude": 1.0},  # Positive charge
        {"position": (0.5, 0.0), "magnitude": -1.0},  # Negative charge
    ]
    
    # Define the computational domain
    domain = dde.geometry.Rectangle([-1, -1], [1, 1])
    
    # Analytical solution for the potential of point charges in 2D
    def analytical_solution(x, y):
        potential = 0
        for charge in charges:
            q = charge["magnitude"]
            x_q, y_q = charge["position"]
            r_squared = (x - x_q)**2 + (y - y_q)**2
            # Avoid singularity
            r_squared = np.maximum(r_squared, 1e-10)
            potential += q * np.log(1.0 / r_squared) / (2 * np.pi * epsilon_0)
        return potential
    
    # Define the PDE: Laplace's equation away from charges
    def pde(x, y):
        """
        ∇²φ = 0 (away from charges)
        """
        phi = y[:, 0:1]
        phi_xx = dde.grad.hessian(phi, x, i=0, j=0)
        phi_yy = dde.grad.hessian(phi, x, i=1, j=1)
        return phi_xx + phi_yy
    
    # Boundary condition: Dirichlet (grounded)
    def boundary_condition(x, on_boundary):
        return on_boundary
    
    def boundary_value(x):
        """
        φ = 0 on the boundary
        """
        return np.zeros((len(x), 1))
    
    # Create the PDE problem
    data = dde.data.PDE(
        geometry=domain,
        pde=pde,
        bcs=[
            dde.DirichletBC(domain, boundary_value, boundary_condition),
        ],
        num_domain=2000,
        num_boundary=200,
    )
    
    # Create the network
    initializer = "Glorot uniform"
    if architecture == "FNN":
        net = dde.nn.FNN([2] + [50] * 4 + [1], activation_func, initializer)
    elif architecture == "ResNet":
        net = dde.nn.ResNet(
            input_size=2,
            output_size=1,
            num_neurons=50,
            num_blocks=2,
            activation=activation_func,
            kernel_initializer=initializer
        )
    elif architecture == "MsFFN":  # Multi-scale FNN
        net = dde.nn.MsFFN(
            layer_sizes=[2] + [50] * 4 + [1],
            activation=activation_func,
            kernel_initializer=initializer,
            sigmas=[0.1, 0.2, 1.0, 5.0]  # Multiple scales
        )
    
    # Create the model
    model = dde.Model(data, net)
    
    # Measure training time
    start_time = time.time()
    
    # Compile and train
    model.compile("adam", lr=lr)
    history = model.train(epochs=num_epochs)
    
    # Fine-tune with L-BFGS
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    
    training_time = time.time() - start_time
    
    # Compute the final errors
    test_x = np.linspace(-1, 1, 100)
    test_y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(test_x, test_y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    points = np.vstack((X_flat, Y_flat)).T
    
    # Compute the PINN solution
    phi_pred = model.predict(points)
    phi_pred = phi_pred.reshape(X.shape)
    
    # Compute the analytical solution
    phi_analytical = np.zeros_like(X)
    for i in range(len(test_x)):
        for j in range(len(test_y)):
            phi_analytical[j, i] = analytical_solution(X[j, i], Y[j, i])
    
    # Compute error metrics
    abs_error = np.abs(phi_pred - phi_analytical)
    mean_abs_error = np.mean(abs_error)
    max_abs_error = np.max(abs_error)
    rmse = np.sqrt(np.mean((phi_pred - phi_analytical) ** 2))
    
    # Return results
    results = {
        "architecture": architecture,
        "activation": activation_func,
        "epochs": num_epochs,
        "learning_rate": lr,
        "training_time": training_time,
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "final_loss": losshistory.loss_train[-1],
        "model": model,
        "phi_pred": phi_pred,
        "phi_analytical": phi_analytical
    }
    
    return results

def main():
    # Create output directory
    output_dir = "architecture_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define experiments to run
    experiments = [
        # Vary architectures with fixed activation
        {"architecture": "FNN", "activation": "tanh", "epochs": 10000},
        {"architecture": "ResNet", "activation": "tanh", "epochs": 10000},
        {"architecture": "MsFFN", "activation": "tanh", "epochs": 10000},
        
        # Vary activations with fixed architecture
        {"architecture": "FNN", "activation": "relu", "epochs": 10000},
        {"architecture": "FNN", "activation": "sin", "epochs": 10000},
        {"architecture": "FNN", "activation": "swish", "epochs": 10000},
    ]
    
    # Run all experiments
    results = []
    for i, exp in enumerate(experiments):
        print(f"\nRunning experiment {i+1}/{len(experiments)}: {exp}")
        result = run_experiment(
            exp["architecture"],
            exp["activation"],
            exp["epochs"]
        )
        results.append(result)
        
        # Create a directory for this experiment
        exp_dir = f"{output_dir}/exp_{i+1}_{exp['architecture']}_{exp['activation']}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save the model performance visualizations
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        
        # Plot the PINN solution
        plt.figure(figsize=(12, 10))
        contour = plt.contourf(X, Y, result["phi_pred"], 50, cmap=cm.viridis)
        plt.colorbar(label='Potential φ')
        plt.title(f'{exp["architecture"]} with {exp["activation"]} - PINN Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{exp_dir}/pinn_solution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot the error
        error = np.abs(result["phi_pred"] - result["phi_analytical"])
        plt.figure(figsize=(12, 10))
        contour = plt.contourf(X, Y, error, 50, cmap=cm.viridis)
        plt.colorbar(label='Absolute Error')
        plt.title(f'{exp["architecture"]} with {exp["activation"]} - Absolute Error')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{exp_dir}/absolute_error.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a dataframe with results for comparison
    results_df = pd.DataFrame([
        {
            "Architecture": r["architecture"],
            "Activation": r["activation"],
            "Training Time (s)": r["training_time"],
            "Mean Abs Error": r["mean_abs_error"],
            "Max Abs Error": r["max_abs_error"],
            "RMSE": r["rmse"],
            "Final Loss": float(r["final_loss"])  # Convert to float here
        }
        for r in results
    ])
    
    # Save results to CSV
    results_df.to_csv(f"{output_dir}/comparison_results.csv", index=False)
    
    # Create bar plots for each metric
    metrics = ["Training Time (s)", "Mean Abs Error", "Max Abs Error", "RMSE", "Final Loss"]
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        
        # Create a unique label for each experiment
        results_df['Experiment'] = results_df['Architecture'] + ' + ' + results_df['Activation']
        
        # Sort by the metric value
        df_sorted = results_df.sort_values(by=metric)
        
        # Create the bar plot
        ax = sns.barplot(x='Experiment', y=metric, data=df_sorted)
        plt.title(f'Comparison of {metric} Across Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add values on top of bars
        for i, v in enumerate(df_sorted[metric]):
            ax.text(i, v + v*0.02, f"{v:.6f}", ha='center')
        
        # Save the figure
        plt.savefig(f"{output_dir}/{metric.lower().replace(' ', '_')}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a heatmap for quick visual comparison
    plt.figure(figsize=(14, 10))
    results_pivot = results_df.pivot_table(
        index="Architecture", 
        columns="Activation", 
        values="RMSE"
    )
    sns.heatmap(results_pivot, annot=True, cmap="YlGnBu", fmt=".6f")
    plt.title('RMSE Comparison Across Architectures and Activations')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print the summary
    print("\n===== Results Summary =====")
    print(results_df.to_string(index=False))
    print("\nBest model by RMSE:", results_df.loc[results_df['RMSE'].idxmin()][['Architecture', 'Activation']])
    
    # Save the summary to text file
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("===== Results Summary =====\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nBest model by RMSE: ")
        f.write(str(results_df.loc[results_df['RMSE'].idxmin()][['Architecture', 'Activation']].to_dict()))

if __name__ == "__main__":
    main()