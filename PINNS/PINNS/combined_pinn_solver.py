import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors  # Add this import for LogNorm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import time
import os
import glob
import sys

# Enable TensorFlow compatibility mode
tf.compat.v1.disable_eager_execution()

# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

# Base directory is the current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_existing_models():
    """Find existing model checkpoint files in the directory"""
    # Look for checkpoint files in the main directory
    checkpoint_files = glob.glob(os.path.join(BASE_DIR, "poisson_singular_model-*.ckpt.index"))
    models = []
    for f in checkpoint_files:
        # Extract the model step number
        step = int(os.path.basename(f).split("-")[1].split(".")[0])
        base_path = f[:-6]  # Remove ".index"
        models.append({"step": step, "path": base_path})
    
    # Sort by step number (descending)
    models.sort(key=lambda x: x["step"], reverse=True)
    return models

def main():
    # Problem parameters
    epsilon_0 = 1.0  # Permittivity of free space (scaled for simplicity)
    
    # Define the charges (position and magnitude)
    charges = [
        {"position": (-0.5, 0.0), "magnitude": 1.0},  # Positive charge
        {"position": (0.5, 0.0), "magnitude": -1.0},  # Negative charge
    ]
    
    # Define the computational domain
    domain = dde.geometry.Rectangle([-1, -1], [1, 1])
    
    # Define the PDE: Poisson's equation with singular source terms
    def pde(x, y):
        """
        ∇²φ = -ρ/ε₀
        
        We'll use a smoother approximation of the delta function
        to represent the singular source terms
        """
        phi = y[:, 0:1]
        phi_xx = dde.grad.hessian(phi, x, i=0, j=0)
        phi_yy = dde.grad.hessian(phi, x, i=1, j=1)
        laplacian = phi_xx + phi_yy
        
        # Source term: approximate delta functions for point charges
        source = 0
        sigma = 0.05  # Width parameter for the smoothed delta function
        
        for charge in charges:
            q = charge["magnitude"]
            x_q, y_q = charge["position"]
            
            # Smoothed delta function (Gaussian approximation)
            distance_squared = (x[:, 0:1] - x_q)**2 + (x[:, 1:2] - y_q)**2
            delta_approx = tf.exp(-distance_squared / (2 * sigma**2)) / (2 * np.pi * sigma**2)
            
            # Add charge contribution
            source += q * delta_approx / epsilon_0
        
        # Poisson's equation
        return laplacian + source
    
    # Boundary condition: Dirichlet (grounded)
    def boundary_condition(x, on_boundary):
        return on_boundary
    
    def boundary_value(x):
        """
        φ = 0 on the boundary
        """
        return np.zeros((len(x), 1))
    
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
    
    # Create the PDE problem
    data = dde.data.PDE(
        geometry=domain,
        pde=pde,
        bcs=[
            dde.DirichletBC(domain, boundary_value, boundary_condition),
        ],
        num_domain=5000,  # More points to better capture the singularities
        num_boundary=200,
    )
    
    # Define the neural network architecture
    layer_size = [2] + [64] * 5 + [1]  # Deeper network for this challenging problem
    activation = "tanh"
    initializer = "Glorot uniform"
    
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    # Create the model
    model = dde.Model(data, net)
    
    # Find existing models
    existing_models = find_existing_models()
    
    if existing_models:
        print("\nFound existing model(s):")
        for i, m in enumerate(existing_models):
            print(f"{i+1}. Model at step {m['step']}: {os.path.basename(m['path'])}")
        
        print("\nWhat would you like to do?")
        print("1. Train a new model")
        print("2. Load an existing model")
        choice = input("Enter your choice (1/2): ")
        
        if choice == "2":
            if len(existing_models) > 1:
                model_idx = input(f"Which model would you like to load (1-{len(existing_models)})? ")
                try:
                    model_idx = int(model_idx) - 1
                    if not 0 <= model_idx < len(existing_models):
                        print("Invalid selection. Using the latest model.")
                        model_idx = 0
                except ValueError:
                    print("Invalid input. Using the latest model.")
                    model_idx = 0
            else:
                model_idx = 0
                
            model_path = existing_models[model_idx]["path"]
            try:
                print(f"Loading model from: {model_path}")
                # Compile first
                model.compile("adam", lr=0.001)
                model.restore(model_path)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Training a new model...")
                choice = "1"
        else:
            choice = "1"
    else:
        print("\nNo existing models found.")
        choice = "1"
    
    if choice == "1":
        print("Training the model. This may take a while...")
        # Compile and train the model
        model.compile("adam", lr=0.001)
        model.train(epochs=30000)
        
        # Fine-tune with L-BFGS
        model.compile("L-BFGS")
        losshistory, train_state = model.train()
        
        # Plot the loss history
        dde.utils.plot_loss_history(losshistory)
        plt.savefig('poisson_loss_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save the model
        model_save_path = os.path.join(BASE_DIR, "poisson_singular_model")
        print(f"Saving model to: {model_save_path}")
        try:
            model.save(model_save_path)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    # Run basic visualization
    plot_results(model, charges, analytical_solution)
    plot_electric_field(model, charges)
    
    # Ask user which advanced visualization to run
    print("\nAdvanced Visualizations:")
    print("1. 3D Electric Field Visualization")
    print("2. Enhanced 3D Visualization with Interactive Elements")
    print("3. Cross-section Analysis")
    print("4. Convergence Analysis (will train multiple models)")
    print("5. Animate Training Progress (will train a new model)")
    print("6. Run All Visualizations")
    print("7. Exit")
    
    vis_choice = input("Choose a visualization (1-7): ")
    
    if vis_choice == "1" or vis_choice == "6":
        visualize_electric_field_3d(model, charges)
    
    if vis_choice == "2" or vis_choice == "6":
        enhanced_3d_visualization(model, charges)
    
    if vis_choice == "3" or vis_choice == "6":
        cross_section_analysis(model, charges)
    
    if vis_choice == "4" or vis_choice == "6":
        convergence_analysis(domain, pde, boundary_condition, boundary_value, charges)
    
    if vis_choice == "5" or vis_choice == "6":
        animate_training_progress(domain, pde, boundary_condition, boundary_value, charges)

def plot_results(model, charges, analytical_solution):
    """
    Plots the PINN solution, analytical solution, and error.
    """
    # Generate a grid for plotting
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    points = np.vstack((X_flat, Y_flat)).T
    
    # Compute the PINN solution
    phi_pred = model.predict(points)
    phi_pred = phi_pred.reshape(X.shape)
    
    # Compute the analytical solution
    phi_analytical = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            phi_analytical[j, i] = analytical_solution(X[j, i], Y[j, i])
    
    # Compute the error
    error = np.abs(phi_pred - phi_analytical)
    
    # Create figure and subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot the PINN solution - 2D contour
    ax1 = fig.add_subplot(231)
    contour1 = ax1.contourf(X, Y, phi_pred, 50, cmap=cm.viridis)
    ax1.set_title('PINN Solution (Contour)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(contour1, ax=ax1)
    
    # Mark the charges
    for charge in charges:
        x_q, y_q = charge["position"]
        q = charge["magnitude"]
        if q > 0:
            ax1.scatter(x_q, y_q, color='red', s=100, marker='+', label='Positive Charge')
        else:
            ax1.scatter(x_q, y_q, color='blue', s=100, marker='o', label='Negative Charge')
    
    # Plot the analytical solution - 2D contour
    ax2 = fig.add_subplot(232)
    contour2 = ax2.contourf(X, Y, phi_analytical, 50, cmap=cm.viridis)
    ax2.set_title('Analytical Solution (Contour)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(contour2, ax=ax2)
    
    # Mark the charges
    for charge in charges:
        x_q, y_q = charge["position"]
        q = charge["magnitude"]
        if q > 0:
            ax2.scatter(x_q, y_q, color='red', s=100, marker='+')
        else:
            ax2.scatter(x_q, y_q, color='blue', s=100, marker='o')
    
    # Plot the error - 2D contour
    ax3 = fig.add_subplot(233)
    contour3 = ax3.contourf(X, Y, error, 50, cmap=cm.viridis)
    ax3.set_title('Absolute Error (Contour)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(contour3, ax=ax3)
    
    # Plot the PINN solution - 3D surface
    ax4 = fig.add_subplot(234, projection='3d')
    surf1 = ax4.plot_surface(X, Y, phi_pred, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax4.set_title('PINN Solution (Surface)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('Potential φ')
    
    # Plot the analytical solution - 3D surface
    ax5 = fig.add_subplot(235, projection='3d')
    surf2 = ax5.plot_surface(X, Y, phi_analytical, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax5.set_title('Analytical Solution (Surface)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('Potential φ')
    
    # Plot the error - 3D surface
    ax6 = fig.add_subplot(236, projection='3d')
    surf3 = ax6.plot_surface(X, Y, error, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax6.set_title('Absolute Error (Surface)')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('Error')
    
    plt.tight_layout()
    plt.savefig('poisson_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate error metrics
    mean_abs_error = np.mean(error)
    max_abs_error = np.max(error)
    rmse = np.sqrt(np.mean((phi_pred - phi_analytical) ** 2))
    
    print(f"Mean absolute error: {mean_abs_error}")
    print(f"Maximum absolute error: {max_abs_error}")
    print(f"RMSE: {rmse}")

def plot_electric_field(model, charges):
    """
    Plots the electric field lines.
    """
    # Generate a grid for plotting
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute the electric field using finite differences
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            # Skip points very close to charges to avoid numerical issues
            too_close = False
            for charge in charges:
                x_q, y_q = charge["position"]
                if np.sqrt((X[j, i] - x_q)**2 + (Y[j, i] - y_q)**2) < 0.05:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Calculate the gradient of the potential at this point
            dx = 0.01
            point = np.array([[X[j, i], Y[j, i]]])
            point_dx_plus = np.array([[X[j, i] + dx, Y[j, i]]])
            point_dy_plus = np.array([[X[j, i], Y[j, i] + dx]])
            
            phi = model.predict(point)[0, 0]
            phi_dx = model.predict(point_dx_plus)[0, 0]
            phi_dy = model.predict(point_dy_plus)[0, 0]
            
            # Electric field is negative gradient of potential
            Ex[j, i] = -(phi_dx - phi) / dx
            Ey[j, i] = -(phi_dy - phi) / dx
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Plot the electric field
    # Normalize the field for better visualization
    E_magnitude = np.sqrt(Ex**2 + Ey**2)
    Ex_norm = Ex / (E_magnitude + 1e-10)
    Ey_norm = Ey / (E_magnitude + 1e-10)
    
    # Skip some points for better visualization
    skip = 5
    
    # Plot the streamlines
    plt.streamplot(X[::skip, ::skip], Y[::skip, ::skip], 
                  Ex_norm[::skip, ::skip], Ey_norm[::skip, ::skip], 
                  color=E_magnitude[::skip, ::skip], 
                  cmap=plt.cm.plasma,
                  density=2,
                  linewidth=1.5,
                  arrowsize=1.5)
    
    # Mark the charges
    for charge in charges:
        x_q, y_q = charge["position"]
        q = charge["magnitude"]
        if q > 0:
            plt.scatter(x_q, y_q, color='red', s=100, marker='+', label='Positive Charge')
        else:
            plt.scatter(x_q, y_q, color='blue', s=100, marker='o', label='Negative Charge')
    
    plt.colorbar(label='Electric Field Magnitude')
    plt.title('Electric Field for Multiple Point Charges (Poisson with Singular Sources)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('poisson_electric_field.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_electric_field_3d(model, charges):
    """
    Creates an advanced 3D visualization of the electric field vectors.
    
    Args:
        model: Trained DeepXDE model
        charges: List of charge dictionaries with position and magnitude
    """
    # Create a 3D grid of points for field visualization
    # Using fewer points for clearer visualization
    nx, ny, nz = 15, 15, 10  # Adjust these numbers based on visualization needs
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(0, 0.5, nz)  # Height above the xy-plane
    
    # Starting points for field lines
    field_line_starts = []
    
    # Create starting points in a circular pattern around each charge
    for charge in charges:
        x_q, y_q = charge["position"]
        radius = 0.15  # Radius of the circle around charge
        n_points = 8  # Number of points around each charge
        
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            start_x = x_q + radius * np.cos(angle)
            start_y = y_q + radius * np.sin(angle)
            field_line_starts.append((start_x, start_y, 0.05))  # Slightly above xy-plane
    
    # Add some random starting points
    n_random = 10
    np.random.seed(42)  # For reproducibility
    for _ in range(n_random):
        rand_x = np.random.uniform(-0.9, 0.9)
        rand_y = np.random.uniform(-0.9, 0.9)
        field_line_starts.append((rand_x, rand_y, 0.05))
    
    # Calculate potential values on a 2D grid for the base contour plot
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # Z=0 plane
    
    points_2d = np.vstack((X.flatten(), Y.flatten())).T
    phi_values = model.predict(points_2d).reshape(X.shape)
    
    # Create the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the base potential contour
    contour = ax.contourf(X, Y, phi_values, 20, zdir='z', offset=0, cmap=cm.viridis, alpha=0.8)
    
    # Calculate and plot electric field vectors
    # Create 3D grid for vector field
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Calculate Electric Field at each point (simplified for better visualization)
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    Ez = np.zeros_like(Z)  # Will be mostly zero in our 2D potential case
    
    # Finite difference step
    dx = 0.01
    
    # Calculate field for each point
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                # Skip points very close to charges
                too_close = False
                for charge in charges:
                    x_q, y_q = charge["position"]
                    if np.sqrt((X[j, i, k] - x_q)**2 + (Y[j, i, k] - y_q)**2) < 0.1:
                        too_close = True
                        break
                
                if too_close:
                    continue
                
                # Calculate field using finite differences
                point = np.array([[X[j, i, k], Y[j, i, k]]])
                point_dx_plus = np.array([[X[j, i, k] + dx, Y[j, i, k]]])
                point_dy_plus = np.array([[X[j, i, k], Y[j, i, k] + dx]])
                
                phi = model.predict(point)[0, 0]
                phi_dx = model.predict(point_dx_plus)[0, 0]
                phi_dy = model.predict(point_dy_plus)[0, 0]
                
                # Electric field is negative gradient of potential
                Ex[j, i, k] = -(phi_dx - phi) / dx
                Ey[j, i, k] = -(phi_dy - phi) / dx
                # Ez is approximately 0 for our 2D potential
    
    # Compute field magnitude
    E_magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    
    # Normalize field for visualization
    Ex_norm = Ex / (E_magnitude + 1e-10)
    Ey_norm = Ey / (E_magnitude + 1e-10)
    Ez_norm = Ez / (E_magnitude + 1e-10)
    
    # Sample the field for clearer visualization
    sample = 2  # Adjust this for different densities
    
    # Plot electric field vectors
    ax.quiver(X[::sample, ::sample, ::sample], 
              Y[::sample, ::sample, ::sample], 
              Z[::sample, ::sample, ::sample],
              Ex_norm[::sample, ::sample, ::sample],
              Ey_norm[::sample, ::sample, ::sample],
              Ez_norm[::sample, ::sample, ::sample],
              length=0.1, normalize=True, color='black', alpha=0.6)
    
    # Draw field lines
    max_steps = 200
    step_size = 0.02
    
    for start in field_line_starts:
        x_line = [start[0]]
        y_line = [start[1]]
        z_line = [start[2]]
        
        for _ in range(max_steps):
            # Get current position
            x_current, y_current, z_current = x_line[-1], y_line[-1], z_line[-1]
            
            # Check if we've left the domain or gotten too close to a charge
            if (abs(x_current) > 1.0 or abs(y_current) > 1.0 or z_current < 0 or z_current > 1.0):
                break
            
            # Check if too close to any charge
            too_close = False
            for charge in charges:
                x_q, y_q = charge["position"]
                dist = np.sqrt((x_current - x_q)**2 + (y_current - y_q)**2)
                if dist < 0.1:  # Minimum safe distance
                    too_close = True
                    break
            
            if too_close:
                break
            
            # Interpolate the field at current position
            # Find nearest grid points for simplicity
            i = np.argmin(np.abs(x - x_current))
            j = np.argmin(np.abs(y - y_current))
            k = np.argmin(np.abs(z - z_current))
            
            # Extract field direction (assuming we're within bounds)
            try:
                ex = Ex_norm[j, i, k]
                ey = Ey_norm[j, i, k]
                ez = Ez_norm[j, i, k]
            except IndexError:
                break
                
            # For positive charges, field lines go outward
            # For negative charges, field lines go inward
            # Determine the sign based on which charge is closest
            closest_charge_dist = float('inf')
            closest_charge_sign = 1
            
            for charge in charges:
                x_q, y_q = charge["position"]
                dist = np.sqrt((x_current - x_q)**2 + (y_current - y_q)**2)
                if dist < closest_charge_dist:
                    closest_charge_dist = dist
                    closest_charge_sign = 1 if charge["magnitude"] > 0 else -1
            
            # Update position using field direction
            sign = closest_charge_sign
            x_next = x_current + sign * ex * step_size
            y_next = y_current + sign * ey * step_size
            z_next = z_current + sign * ez * step_size
            
            # Add new point to the line
            x_line.append(x_next)
            y_line.append(y_next)
            z_line.append(z_next)
        
        # Plot the field line
        ax.plot(x_line, y_line, z_line, '-', linewidth=1.5, alpha=0.7, 
               color=plt.cm.coolwarm(0.8 if len(x_line) > max_steps/2 else 0.2))
    
    # Mark the charges
    for charge in charges:
        x_q, y_q = charge["position"]
        q = charge["magnitude"]
        if q > 0:
            ax.scatter([x_q], [y_q], [0], color='red', s=200, marker='+', linewidth=3, label='Positive Charge')
        else:
            ax.scatter([x_q], [y_q], [0], color='blue', s=200, marker='o', label='Negative Charge')
    
    # Add a colorbar for the potential
    cbar = fig.colorbar(contour, ax=ax, shrink=0.7)
    cbar.set_label('Electric Potential φ')
    
    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Electric Field Visualization')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,0.5])
    
    # Add a legend (only once for each type)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.savefig('electric_field_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def enhanced_3d_visualization(model, charges):
    """
    Creates an enhanced 3D visualization of the electrostatic potential with 
    interactive elements and multiple views.
    
    Args:
        model: Trained DeepXDE model
        charges: List of charge dictionaries with position and magnitude
    """
    # Higher resolution for smoother visualization
    x = np.linspace(-1, 1, 120)
    y = np.linspace(-1, 1, 120)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    points = np.vstack((X_flat, Y_flat)).T
    
    # Compute the PINN solution
    phi_pred = model.predict(points)
    phi_pred = phi_pred.reshape(X.shape)
    
    # Compute the electric field using finite differences
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    dx = 0.01
    
    for i in range(len(x)):
        for j in range(len(y)):
            # Skip points very close to charges
            too_close = False
            for charge in charges:
                x_q, y_q = charge["position"]
                if np.sqrt((X[j, i] - x_q)**2 + (Y[j, i] - y_q)**2) < 0.05:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Calculate the gradient of the potential
            point = np.array([[X[j, i], Y[j, i]]])
            point_dx_plus = np.array([[X[j, i] + dx, Y[j, i]]])
            point_dy_plus = np.array([[X[j, i], Y[j, i] + dx]])
            
            phi = model.predict(point)[0, 0]
            phi_dx = model.predict(point_dx_plus)[0, 0]
            phi_dy = model.predict(point_dy_plus)[0, 0]
            
            # Electric field is negative gradient of potential
            Ex[j, i] = -(phi_dx - phi) / dx
            Ey[j, i] = -(phi_dy - phi) / dx
    
    # Create a figure with 2 subplots, one for potential and one for electric field
    fig = plt.figure(figsize=(20, 10))
    
    # 3D surface plot with custom perspective
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create a normalization instance for consistent color mapping
    norm = colors.Normalize(vmin=np.min(phi_pred), vmax=np.max(phi_pred))
    
    # Initial surface plot
    surf = ax1.plot_surface(X, Y, phi_pred, cmap=cm.viridis, linewidth=0, 
                           antialiased=True, norm=norm)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)
    cbar.set_label('Potential φ')
    
    # Add equipotential contours beneath the surface
    offset = np.min(phi_pred) - 0.1 * np.abs(np.min(phi_pred))
    cset = ax1.contour(X, Y, phi_pred, zdir='z', offset=offset, 
                      levels=15, cmap=cm.viridis)
    
    # Mark the charges in 3D
    for charge in charges:
        x_q, y_q = charge["position"]
        q = charge["magnitude"]
        if q > 0:
            ax1.scatter([x_q], [y_q], [np.max(phi_pred) * 1.1], 
                      color='red', s=100, marker='+', linewidth=3)
        else:
            ax1.scatter([x_q], [y_q], [np.min(phi_pred) * 1.1], 
                      color='blue', s=100, marker='o')
    
    # Set limits and labels
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Potential φ')
    ax1.set_title('Electrostatic Potential (3D Surface)')
    
    # Electric field visualization with streamlines and potential contours
    ax2 = fig.add_subplot(122)
    
    # Calculate field magnitude
    E_magnitude = np.sqrt(Ex**2 + Ey**2)
    
    # Create a logarithmic colormap to better visualize the field strength variations
    norm = colors.LogNorm(vmin=max(E_magnitude.min(), 1e-3), vmax=E_magnitude.max())
    
    # Plot the potential contours
    contours = ax2.contour(X, Y, phi_pred, 20, cmap='viridis', alpha=0.7)
    plt.colorbar(contours, ax=ax2, label='Potential φ', shrink=0.6)
    
    # Plot streamlines with varying width according to field strength
    streamlines = ax2.streamplot(X, Y, Ex, Ey, density=2, color='black', 
                              linewidth=2*E_magnitude/E_magnitude.max())
    
    # Mark the charges
    for charge in charges:
        x_q, y_q = charge["position"]
        q = charge["magnitude"]
        if q > 0:
            ax2.scatter(x_q, y_q, color='red', s=150, marker='+', linewidth=3, 
                      label='Positive Charge')
        else:
            ax2.scatter(x_q, y_q, color='blue', s=150, marker='o', 
                      label='Negative Charge')
    
    ax2.set_title('Electric Field and Equipotential Lines')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.legend()
    
    # Add sliders for interactive adjustment of the view
    plt.subplots_adjust(bottom=0.25)
    
    ax_elev = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_azim = plt.axes([0.25, 0.15, 0.65, 0.03])
    
    slider_elev = Slider(ax_elev, 'Elevation', 0.0, 90.0, valinit=30)
    slider_azim = Slider(ax_azim, 'Azimuth', 0.0, 360.0, valinit=30)
    
    def update(val):
        ax1.view_init(elev=slider_elev.val, azim=slider_azim.val)
        fig.canvas.draw_idle()
    
    slider_elev.on_changed(update)
    slider_azim.on_changed(update)
    
    plt.tight_layout()
    
    # Create animation of rotating view
    def animate(i):
        ax1.view_init(elev=30, azim=i)
        return [surf]
    
    # Uncomment to save the animation
    # ani = animation.FuncAnimation(fig, animate, frames=360, interval=50, blit=True)
    # ani.save('potential_3d_rotation.mp4', writer='ffmpeg', fps=30)
    
    plt.savefig('enhanced_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def cross_section_analysis(model, charges):
    """
    Creates cross-section analyses along different axes through the charges.
    
    Args:
        model: Trained DeepXDE model
        charges: List of charge dictionaries with position and magnitude
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cross-section along x-axis through y=0 (passing through both charges)
    x_line = np.linspace(-1, 1, 500)
    y_const = np.zeros_like(x_line)
    points_x = np.vstack((x_line, y_const)).T
    
    # Compute the PINN solution
    phi_x_pred = model.predict(points_x)
    
    # Compute analytical solution
    phi_x_analytical = np.zeros_like(x_line)
    for i, x in enumerate(x_line):
        phi = 0
        for charge in charges:
            q = charge["magnitude"]
            x_q, y_q = charge["position"]
            r_squared = (x - x_q)**2 + (0 - y_q)**2
            r_squared = max(r_squared, 1e-10)  # Avoid singularity
            phi += q * np.log(1.0 / r_squared) / (2 * np.pi)
        phi_x_analytical[i] = phi
    
    # Plot cross-section along x-axis
    axes[0, 0].plot(x_line, phi_x_pred, 'b-', linewidth=3, label='PINN Solution')
    axes[0, 0].plot(x_line, phi_x_analytical, 'r--', linewidth=2, label='Analytical')
    
    # Mark charge positions
    for charge in charges:
        x_q, y_q = charge["position"]
        if y_q == 0:  # only mark charges on this line
            axes[0, 0].axvline(x=x_q, color='gray', linestyle=':', alpha=0.7)
            
            if charge["magnitude"] > 0:
                axes[0, 0].scatter(x_q, 0, color='red', s=100, marker='+')
            else:
                axes[0, 0].scatter(x_q, 0, color='blue', s=100, marker='o')
    
    axes[0, 0].set_title('Potential Along x-axis (y=0)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Potential φ')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Cross-section along y-axis through center (x=0)
    y_line = np.linspace(-1, 1, 500)
    x_const = np.zeros_like(y_line)
    points_y = np.vstack((x_const, y_line)).T
    
    # Compute the PINN solution
    phi_y_pred = model.predict(points_y)
    
    # Compute analytical solution
    phi_y_analytical = np.zeros_like(y_line)
    for i, y in enumerate(y_line):
        phi = 0
        for charge in charges:
            q = charge["magnitude"]
            x_q, y_q = charge["position"]
            r_squared = (0 - x_q)**2 + (y - y_q)**2
            r_squared = max(r_squared, 1e-10)  # Avoid singularity
            phi += q * np.log(1.0 / r_squared) / (2 * np.pi)
        phi_y_analytical[i] = phi
    
    # Plot cross-section along y-axis
    axes[0, 1].plot(y_line, phi_y_pred, 'b-', linewidth=3, label='PINN Solution')
    axes[0, 1].plot(y_line, phi_y_analytical, 'r--', linewidth=2, label='Analytical')
    axes[0, 1].set_title('Potential Along y-axis (x=0)')
    axes[0, 1].set_xlabel('y')
    axes[0, 1].set_ylabel('Potential φ')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Log-scale error analysis
    axes[1, 0].semilogy(x_line, np.abs(phi_x_pred - phi_x_analytical) + 1e-10, 'g-', linewidth=2)
    axes[1, 0].set_title('Log-Scale Error Along x-axis')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Log(|Error|)')
    axes[1, 0].grid(True)
    
    axes[1, 1].semilogy(y_line, np.abs(phi_y_pred - phi_y_analytical) + 1e-10, 'g-', linewidth=2)
    axes[1, 1].set_title('Log-Scale Error Along y-axis')
    axes[1, 1].set_xlabel('y')
    axes[1, 1].set_ylabel('Log(|Error|)')
    axes[1, 1].grid(True)
    
    # Mark charge positions on error plots as well
    for charge in charges:
        x_q, y_q = charge["position"]
        if y_q == 0:  # only mark charges on this line
            axes[1, 0].axvline(x=x_q, color='gray', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('cross_section_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def convergence_analysis(domain, pde, boundary_condition, boundary_value, charges):
    """
    Performs a convergence analysis with different network architectures and parameters.
    
    Args:
        domain: DeepXDE geometry object
        pde: PDE function
        boundary_condition, boundary_value: Boundary condition functions
        charges: List of charge dictionaries with position and magnitude
    """
    # Define configurations to test
    network_architectures = [
        {"name": "Shallow", "layers": [2, 20, 20, 1]},
        {"name": "Medium", "layers": [2, 40, 40, 40, 1]},
        {"name": "Deep", "layers": [2, 20, 20, 20, 20, 20, 20, 1]},
        {"name": "Wide", "layers": [2, 100, 100, 1]},
    ]
    
    activation_functions = ["tanh", "relu", "sin"]
    
    results = []
    
    # Define a test point grid for evaluation
    x_test = np.linspace(-0.9, 0.9, 50)  # Avoid boundaries
    y_test = np.linspace(-0.9, 0.9, 50)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    X_flat = X_test.flatten()
    Y_flat = Y_test.flatten()
    points_test = np.vstack((X_flat, Y_flat)).T
    
    # Compute reference analytical solution
    phi_analytical = np.zeros(len(points_test))
    for i, point in enumerate(points_test):
        x, y = point
        phi = 0
        for charge in charges:
            q = charge["magnitude"]
            x_q, y_q = charge["position"]
            r_squared = (x - x_q)**2 + (y - y_q)**2
            r_squared = max(r_squared, 1e-10)  # Avoid singularity
            phi += q * np.log(1.0 / r_squared) / (2 * np.pi)
        phi_analytical[i] = phi
    
    # Test each architecture
    for arch in network_architectures:
        for act in activation_functions:
            print(f"Testing {arch['name']} network with {act} activation...")
            
            # Create data
            data = dde.data.PDE(
                geometry=domain,
                pde=pde,
                bcs=[dde.DirichletBC(domain, boundary_value, boundary_condition)],
                num_domain=3000,
                num_boundary=200,
            )
            
            # Create network
            net = dde.nn.FNN(arch["layers"], act, "Glorot uniform")
            
            # Create model
            model = dde.Model(data, net)
            
            # Training settings
            epochs = 5000  # Reduced for comparison purposes
            
            # Start timer
            start_time = time.time()
            
            # Train
            model.compile("adam", lr=0.001)
            model.train(epochs=epochs)
            
            # Finish timer
            train_time = time.time() - start_time
            
            # Predict on test points
            phi_pred = model.predict(points_test)
            
            # Calculate errors
            abs_error = np.abs(phi_pred - phi_analytical.reshape(-1, 1))
            mean_abs_error = np.mean(abs_error)
            max_abs_error = np.max(abs_error)
            rmse = np.sqrt(np.mean((phi_pred - phi_analytical.reshape(-1, 1)) ** 2))
            
            # Store results
            results.append({
                "architecture": arch["name"],
                "activation": act,
                "mean_abs_error": mean_abs_error,
                "max_abs_error": max_abs_error,
                "rmse": rmse,
                "training_time": train_time
            })
            
            print(f"  RMSE: {rmse:.6f}, Training time: {train_time:.2f}s")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for plotting
    arch_names = [r["architecture"] for r in results]
    act_names = [r["activation"] for r in results]
    rmse_values = [r["rmse"] for r in results]
    times = [r["training_time"] for r in results]
    
    # Create labels for x-axis
    labels = [f"{arch}\n{act}" for arch, act in zip(arch_names, act_names)]
    
    # RMSE plot
    axes[0].bar(labels, rmse_values, color='skyblue')
    axes[0].set_title('RMSE by Network Configuration')
    axes[0].set_ylabel('RMSE')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Training time plot
    axes[1].bar(labels, times, color='salmon')
    axes[1].set_title('Training Time by Network Configuration')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print recommendation
    best_idx = rmse_values.index(min(rmse_values))
    print(f"\nRecommended configuration:")
    print(f"Network: {arch_names[best_idx]}")
    print(f"Activation: {act_names[best_idx]}")
    print(f"RMSE: {rmse_values[best_idx]:.6f}")
    print(f"Training time: {times[best_idx]:.2f}s")

def animate_training_progress(domain, pde, boundary_condition, boundary_value, charges):
    """
    Creates an animation showing how the solution evolves during training.
    
    Args:
        domain: DeepXDE geometry object
        pde: PDE function
        boundary_condition, boundary_value: Boundary condition functions
        charges: List of charge dictionaries with position and magnitude
    """
    import time
    
    # Create grid for visualization
    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.flatten(), Y.flatten())).T
    
    # Compute analytical solution for reference
    phi_analytical = np.zeros(len(points))
    for i, point in enumerate(points):
        x, y = point
        phi = 0
        for charge in charges:
            q = charge["magnitude"]
            x_q, y_q = charge["position"]
            r_squared = (x - x_q)**2 + (y - y_q)**2
            r_squared = max(r_squared, 1e-10)  # Avoid singularity
            phi += q * np.log(1.0 / r_squared) / (2 * np.pi)
        phi_analytical[i] = phi
    
    phi_analytical = phi_analytical.reshape(X.shape)
    
    # Create PDE problem with fewer points for faster training
    data = dde.data.PDE(
        geometry=domain,
        pde=pde,
        bcs=[dde.DirichletBC(domain, boundary_value, boundary_condition)],
        num_domain=2000,
        num_boundary=200,
    )
    
    # Define network with fewer layers/neurons for faster training
    layer_size = [2, 20, 20, 20, 1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    # Create the model
    model = dde.Model(data, net)
    
    # Compile with Adam optimizer
    model.compile("adam", lr=0.001)
    
    # Create figure for animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Function to update the plot at each frame
    def update_plot(i):
        # Train for a few epochs
        model.train(epochs=100, display_every=100, disregard_previous_best=True)
        
        # Get current prediction
        phi_pred = model.predict(points).reshape(X.shape)
        
        # Calculate error
        error = np.abs(phi_pred - phi_analytical)
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Plot prediction
        contour1 = ax1.contourf(X, Y, phi_pred, 30, cmap=cm.viridis)
        ax1.set_title(f'PINN Solution (Iteration {i*100})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # Mark charges
        for charge in charges:
            x_q, y_q = charge["position"]
            q = charge["magnitude"]
            if q > 0:
                ax1.scatter(x_q, y_q, color='red', s=100, marker='+')
            else:
                ax1.scatter(x_q, y_q, color='blue', s=100, marker='o')
        
        # Plot error
        contour2 = ax2.contourf(X, Y, error, 30, cmap=cm.viridis)
        ax2.set_title(f'Absolute Error (Iteration {i*100})')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        # Calculate and display metrics
        mean_error = np.mean(error)
        max_error = np.max(error)
        rmse = np.sqrt(np.mean((phi_pred - phi_analytical) ** 2))
        
        ax2.text(0.5, -0.15, f'Mean Error: {mean_error:.6f}, Max Error: {max_error:.6f}, RMSE: {rmse:.6f}',
                horizontalalignment='center', transform=ax2.transAxes)
        
        return contour1, contour2
    
    # Create animation (30 frames = 3000 epochs total)
    ani = animation.FuncAnimation(fig, update_plot, frames=30, interval=200, blit=False)
    
    # Save animation
    ani.save('training_progress.gif', writer='pillow', fps=2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Import additional libraries
    import time
    
    # Start timer
    start_time = time.time()
    
    # Run main function
    main()
    
    # End timer
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)") 