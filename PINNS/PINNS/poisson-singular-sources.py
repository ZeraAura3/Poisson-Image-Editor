import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# Enable TensorFlow compatibility mode
tf.compat.v1.disable_eager_execution()

# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

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
    
    # Visualization
    def plot_results():
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
    
    # Plot electric field
    def plot_electric_field():
        # Generate a grid for plotting
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute the electric field using automatic differentiation
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        points = np.vstack((X_flat, Y_flat)).T
        
        # We'll use finite differences for visualization purposes
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
        plt.close()
    
    # Run the visualizations
    plot_results()
    plot_electric_field()
    
    # Calculate and print error metrics
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
    
    # Calculate error metrics
    abs_error = np.abs(phi_pred - phi_analytical)
    mean_abs_error = np.mean(abs_error)
    max_abs_error = np.max(abs_error)
    rmse = np.sqrt(np.mean((phi_pred - phi_analytical) ** 2))
    
    print(f"Mean absolute error: {mean_abs_error}")
    print(f"Maximum absolute error: {max_abs_error}")
    print(f"RMSE: {rmse}")
    
    # Save the model
    model.save("poisson_singular_model")

if __name__ == "__main__":
    main()