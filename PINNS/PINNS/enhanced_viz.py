def enhanced_3d_visualization(model, charges):
    """
    Creates an enhanced 3D visualization of the electrostatic potential with 
    interactive elements and multiple views.
    
    Args:
        model: Trained DeepXDE model
        charges: List of charge dictionaries with position and magnitude
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider
    
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

# Call this function in your main:
# enhanced_3d_visualization(model, charges)


def cross_section_analysis(model, charges):
    """
    Creates cross-section analyses along different axes through the charges.
    
    Args:
        model: Trained DeepXDE model
        charges: List of charge dictionaries with position and magnitude
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
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

# Call this function in your main:
# cross_section_analysis(model, charges)


def convergence_analysis(domain, pde, boundary_condition, boundary_value, charges):
    """
    Performs a convergence analysis with different network architectures and parameters.
    
    Args:
        domain: DeepXDE geometry object
        pde: PDE function
        boundary_condition, boundary_value: Boundary condition functions
        charges: List of charge dictionaries with position and magnitude
    """
    import deepxde as dde
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
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

# Call this function in your main:
# convergence_analysis(domain, pde, boundary_condition, boundary_value, charges)
