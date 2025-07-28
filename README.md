# Poisson's Equation Applications

This repository contains implementations and applications of Poisson's equation in two distinct domains: **Image Editing** and **Physics-Informed Neural Networks (PINNs)** for electrostatics. The project demonstrates the versatility of Poisson's equation in both computer graphics and computational physics.

## 🖼️ Poisson Image Editing

The Poisson Image Editing component implements seamless image blending techniques using Poisson's equation to create natural-looking composite images.

### Overview

Poisson image editing is a powerful technique for seamless image composition that preserves the gradient field while blending images. This implementation includes:

- **Gradient-domain image editing**
- **Seamless cloning and blending**
- **Interactive and automatic mask creation**
- **Multi-scale processing using Gaussian and Laplacian pyramids**

### Features

- **Interactive Mask Creation**: Draw custom masks for precise control over blending regions
- **Automatic Foreground Extraction**: Intelligent background removal for quick editing
- **Pyramid-based Blending**: Multi-scale processing for better quality results
- **Real-time Visualization**: Preview results during the editing process

### Key Files

```
Poisson Image Editing/
├── image_editing.ipynb          # Main image editing implementation
├── glp.ipynb                    # Gaussian and Laplacian pyramid processing
├── Poisson Image Editing.pdf    # Technical documentation
└── pics/                        # Sample images and results
    ├── source.jpg              # Source images for blending
    ├── target.png              # Target images
    ├── result.jpg              # Final composite results
    └── ...
```

### Usage

1. **Interactive Image Blending**:
   ```python
   # Load the notebook and run cells
   blender = ImageBlender()
   mask = blender.create_mask(source_image, interactive=True)
   result = blender.blend_images(source, target, mask)
   ```

2. **Automatic Processing**:
   ```python
   # Automatic mask generation and blending
   mask = blender.create_mask(source_image, interactive=False)
   result = blender.poisson_blend(source, target, mask)
   ```

### Mathematical Foundation

The Poisson image editing technique solves:

```
∇²f = ∇·v over Ω
f = f* on ∂Ω
```

Where:
- `f` is the unknown function (pixel intensities)
- `v` is the guidance vector field (gradients)
- `Ω` is the region of interest
- `f*` are the boundary conditions

### Requirements

```python
numpy
opencv-python
scipy
matplotlib
jupyter
```

## ⚡ Physics-Informed Neural Networks (PINNs)

The PINNs component solves 2D electrostatic potential problems using deep learning techniques that incorporate physical laws directly into the neural network training process.

### Overview

This implementation uses PINNs to solve Poisson's equation for electrostatic potential with multiple point charges:

```
∇²φ(x,y) = -ρ(x,y)/ε₀
```

Where:
- `φ(x,y)` is the electrostatic potential
- `ρ(x,y)` is the charge density
- `ε₀` is the permittivity of free space

### Features

- **Multiple PINN Architectures**: Compare FNN, ResNet, and MsFFN architectures
- **Activation Function Studies**: Evaluate tanh, ReLU, sin, and swish activations
- **Comprehensive Visualization**: 2D contours, 3D surfaces, and electric field lines
- **Performance Analysis**: Training metrics, convergence studies, and error analysis
- **Configurable Parameters**: Customizable domain size, network architecture, and training parameters

### Key Files

```
PINNS/PINNS/
├── electrostatics-pinn.py           # Main PINN solver
├── electrostatics-config.py         # Configurable PINN implementation  
├── architecture-comparison.py       # Architecture and activation comparison
├── poisson-singular-sources.py      # Advanced singular source handling
├── combined_pinn_solver.py          # Unified solver with multiple features
├── electric_field_3d.py            # 3D electric field visualization
├── enhanced_viz.py                 # Advanced visualization tools
├── activation_comparison_results/   # Activation function study results
├── architecture_comparison_results/ # Architecture comparison results
├── model_checkpoint/               # Trained model checkpoints
└── README.md                       # Detailed PINN documentation
```

### Quick Start

1. **Basic PINN Solver**:
   ```bash
   cd "PINNS/PINNS"
   python electrostatics-pinn.py
   ```

2. **Configurable Solver**:
   ```bash
   python electrostatics-config.py --domain_size 2.0 --epochs 20000 --hidden_layers 4 --neurons 50
   ```

3. **Architecture Comparison**:
   ```bash
   python architecture-comparison.py
   ```

4. **Advanced Singular Sources**:
   ```bash
   python poisson-singular-sources.py
   ```

### Results and Visualizations

The implementation generates:
- **Potential field contours** and 3D surface plots
- **Electric field vector plots** and streamlines  
- **Error analysis** comparing PINN vs analytical solutions
- **Training convergence** plots and metrics
- **Comparative studies** of different architectures and activations

### Mathematical Background

**Analytical Solution for Point Charges:**
```
φ_analytical(x,y) = Σᵢ (qᵢ/(2πε₀)) ln(1/√((x-xᵢ)² + (y-yᵢ)²))
```

**Electric Field:**
```
E = -∇φ
```

**PINN Loss Function:**
```
Loss = Loss_PDE + Loss_BC + Loss_data
```

### Requirements

```python
deepxde
tensorflow
numpy
matplotlib
pandas
seaborn
scipy
```

## 📚 Documentation

- **Poisson Image Editing.pdf**: Complete technical documentation for image editing
- **pinns report.pdf**: Comprehensive report on PINN implementation and results
- **Poisson's Equation Application.pptx**: Project presentation slides

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "College_Assignment/PDE/Project"
   ```

2. **Install Python dependencies**:
   ```bash
   # For Image Editing
   pip install numpy opencv-python scipy matplotlib jupyter
   
   # For PINNs  
   pip install deepxde tensorflow numpy matplotlib pandas seaborn
   ```

3. **Launch Jupyter notebooks** (for image editing):
   ```bash
   cd "Poisson Image Editing"
   jupyter notebook
   ```

## 🎯 Applications

### Image Editing Applications
- **Photo compositing** and seamless object insertion
- **Background replacement** with natural blending
- **Texture synthesis** and image inpainting
- **Digital art creation** and photo manipulation

### PINNs Applications
- **Electrostatic field analysis** in engineering design
- **Capacitor and conductor simulations**
- **Educational demonstrations** of electromagnetic theory
- **Research in physics-informed machine learning**

## 🔬 Research Contributions

This project demonstrates:
1. **Comparative analysis** of PINN architectures for electrostatics
2. **Activation function impact** on physics-informed learning
3. **Practical implementation** of gradient-domain image editing
4. **Cross-domain applications** of Poisson's equation

## 📈 Performance Metrics

The PINNs implementation achieves:
- **High accuracy**: RMSE < 0.01 for most configurations
- **Fast convergence**: Typical training in 10,000-30,000 epochs
- **Robust performance**: Consistent results across different architectures

## 🤝 Contributing

Feel free to contribute by:
- Adding new PINN architectures or activation functions
- Implementing 3D extensions
- Improving image editing algorithms
- Enhancing visualization capabilities

## 📜 License

This project is part of an academic assignment. Please check with the authors for usage permissions.

## 📞 Contact

For questions or collaborations, please contact the project authors.

---

**Note**: This repository demonstrates the mathematical elegance and practical utility of Poisson's equation across multiple disciplines, from computer graphics to computational physics.
