def visualize_electric_field_3d(model, charges):
    """
    Creates an advanced 3D visualization of the electric field vectors.
    
    Args:
        model: Trained DeepXDE model
        charges: List of charge dictionaries with position and magnitude
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    
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

# Call this function in your main:
# visualize_electric_field_3d(model, charges)


def animate_training_progress(domain, pde, boundary_condition, boundary_value, charges):
    """
    Creates an animation showing how the solution evolves during training.
    
    Args:
        domain: DeepXDE geometry object
        pde: PDE function
        boundary_condition, boundary_value: Boundary condition functions
        charges: List of charge dictionaries with position and magnitude
    """
    import deepxde as dde
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm
    import tensorflow as tf
    
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

# Call this function in your main:
# animate_training_progress(domain, pde, boundary_condition, boundary_value, charges)
