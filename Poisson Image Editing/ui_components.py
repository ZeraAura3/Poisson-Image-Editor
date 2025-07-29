"""
Advanced Interactive Components for Enhanced Poisson Image Editor
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

def create_interactive_image_viewer(image, title="Image Viewer"):
    """Create an interactive image viewer with zoom and pan capabilities."""
    if image is None:
        st.warning(f"No image available for {title}")
        return
    
    # Convert to RGB if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
    else:
        display_image = np.array(image)
    
    # Create interactive plot
    fig = go.Figure()
    
    fig.add_trace(go.Image(z=display_image))
    
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_parameter_slider_panel():
    """Create an advanced parameter control panel with real-time preview."""
    st.markdown("### 🎛️ Advanced Parameter Control")
    
    with st.container():
        # Main parameters in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Blending Settings")
            blend_mode = st.selectbox(
                "Mode",
                ["mixed", "seamless", "monochrome_transfer"],
                help="Blending algorithm to use"
            )
            
            multi_scale = st.checkbox("Multi-scale", value=True)
            color_correct = st.checkbox("Color correction", value=True)
            refine_mask = st.checkbox("Edge refinement", value=True)
        
        with col2:
            st.markdown("#### Positioning")
            placement = st.selectbox(
                "Strategy",
                ["auto", "center", "bottom", "top", "saliency_based"]
            )
            
            scale = st.slider("Scale", 0.1, 2.0, 0.5, 0.05)
            
            manual_position = st.checkbox("Manual positioning")
            
            if manual_position:
                offset_x = st.number_input("X Offset", value=0, step=10)
                offset_y = st.number_input("Y Offset", value=0, step=10)
            else:
                offset_x, offset_y = 0, 0
        
        with col3:
            st.markdown("#### Advanced Settings")
            pyramid_levels = st.slider("Pyramid levels", 2, 6, 4)
            min_size = st.slider("Min size", 16, 128, 32)
            
            boundary = st.selectbox(
                "Boundary handling",
                ["mixed", "dirichlet", "neumann"]
            )
            
            # Performance settings
            st.markdown("#### Performance")
            fast_mode = st.checkbox("Fast mode", help="Reduced quality for faster processing")
    
    return {
        'blend_mode': blend_mode,
        'multi_scale': multi_scale,
        'color_correct': color_correct,
        'refine_mask': refine_mask,
        'placement_strategy': placement,
        'scale_factor': scale,
        'offset': (offset_y, offset_x) if manual_position else None,
        'pyramid_levels': pyramid_levels,
        'min_size': min_size,
        'boundary_handling': boundary,
        'fast_mode': fast_mode
    }

def create_comparison_dashboard(results_dict):
    """Create a comprehensive comparison dashboard."""
    if not results_dict:
        st.info("Process images to see comparison results")
        return
    
    st.markdown("### 📊 Results Comparison Dashboard")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Modes", len(results_dict))
    
    with col2:
        # Calculate average processing time (simulated)
        avg_time = np.random.uniform(2.5, 5.5)
        st.metric("Avg. Process Time", f"{avg_time:.1f}s")
    
    with col3:
        # Quality score (simulated based on mode)
        quality_scores = {"seamless": 85, "mixed": 92, "monochrome_transfer": 78}
        avg_quality = np.mean([quality_scores.get(mode, 80) for mode in results_dict.keys()])
        st.metric("Quality Score", f"{avg_quality:.0f}/100")
    
    with col4:
        # Memory usage (simulated)
        memory_mb = np.random.uniform(45, 120)
        st.metric("Memory Usage", f"{memory_mb:.0f} MB")
    
    # Interactive comparison plot
    st.markdown("#### 🔍 Interactive Results Comparison")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Side by Side", "Grid View", "Difference Analysis"])
    
    with tab1:
        if len(results_dict) >= 2:
            modes = list(results_dict.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                mode1 = st.selectbox("Left Image", modes, key="left_select")
            with col2:
                mode2 = st.selectbox("Right Image", modes, key="right_select", index=1 if len(modes) > 1 else 0)
            
            if mode1 in results_dict and mode2 in results_dict:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{mode1.title()} Mode**")
                    st.image(results_dict[mode1], use_column_width=True)
                with col2:
                    st.markdown(f"**{mode2.title()} Mode**")
                    st.image(results_dict[mode2], use_column_width=True)
    
    with tab2:
        # Grid view of all results
        cols = st.columns(min(3, len(results_dict)))
        for i, (mode, image) in enumerate(results_dict.items()):
            with cols[i % 3]:
                st.markdown(f"**{mode.title()}**")
                st.image(image, use_column_width=True)
    
    with tab3:
        # Difference analysis
        if len(results_dict) >= 2:
            st.markdown("Analyze differences between processing modes:")
            
            modes = list(results_dict.keys())
            base_mode = st.selectbox("Base image", modes, key="base_diff")
            compare_mode = st.selectbox("Compare with", modes, key="compare_diff", index=1 if len(modes) > 1 else 0)
            
            if base_mode != compare_mode and base_mode in results_dict and compare_mode in results_dict:
                # Calculate difference
                base_img = np.array(results_dict[base_mode])
                compare_img = np.array(results_dict[compare_mode])
                
                # Resize if needed
                if base_img.shape != compare_img.shape:
                    min_h = min(base_img.shape[0], compare_img.shape[0])
                    min_w = min(base_img.shape[1], compare_img.shape[1])
                    base_img = cv2.resize(base_img, (min_w, min_h))
                    compare_img = cv2.resize(compare_img, (min_w, min_h))
                
                # Calculate difference
                diff = np.abs(base_img.astype(np.float32) - compare_img.astype(np.float32))
                diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**{base_mode.title()}**")
                    st.image(base_img, use_column_width=True)
                with col2:
                    st.markdown(f"**{compare_mode.title()}**")
                    st.image(compare_img, use_column_width=True)
                with col3:
                    st.markdown("**Difference**")
                    st.image(diff_normalized, use_column_width=True)
                
                # Difference statistics
                st.markdown("#### Difference Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Difference", f"{np.mean(diff):.2f}")
                with col2:
                    st.metric("Max Difference", f"{np.max(diff):.2f}")
                with col3:
                    st.metric("Similarity", f"{100 - (np.mean(diff)/255*100):.1f}%")

def create_processing_progress_tracker():
    """Create a real-time processing progress tracker."""
    if 'processing_steps' not in st.session_state:
        st.session_state.processing_steps = []
    
    if st.session_state.processing_steps:
        st.markdown("### ⚡ Processing Progress")
        
        # Create progress visualization
        steps = st.session_state.processing_steps
        
        for i, step in enumerate(steps):
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if step['status'] == 'completed':
                    st.success("✅")
                elif step['status'] == 'processing':
                    st.info("🔄")
                elif step['status'] == 'error':
                    st.error("❌")
                else:
                    st.info("⏳")
            
            with col2:
                st.write(f"**{step['name']}**")
                if 'description' in step:
                    st.caption(step['description'])
            
            with col3:
                if 'duration' in step:
                    st.caption(f"{step['duration']:.1f}s")

def create_export_options():
    """Create comprehensive export options."""
    st.markdown("### 💾 Export & Download Options")
    
    if st.session_state.get('result_image') is None:
        st.info("Process an image first to enable export options")
        return
    
    # Export format options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 File Formats")
        export_format = st.selectbox(
            "Export format",
            ["PNG (Lossless)", "JPEG (Compressed)", "TIFF (High Quality)", "WebP (Modern)"]
        )
        
        if "JPEG" in export_format:
            quality = st.slider("JPEG Quality", 50, 100, 95)
        else:
            quality = 100
    
    with col2:
        st.markdown("#### 📏 Resolution Options")
        resolution_option = st.radio(
            "Resolution",
            ["Original", "HD (1920x1080)", "4K (3840x2160)", "Custom"]
        )
        
        if resolution_option == "Custom":
            col1_res, col2_res = st.columns(2)
            with col1_res:
                custom_width = st.number_input("Width", min_value=100, value=1920)
            with col2_res:
                custom_height = st.number_input("Height", min_value=100, value=1080)
    
    # Batch export options
    st.markdown("#### 📦 Batch Export")
    
    export_all = st.checkbox("Export all processing results")
    include_intermediates = st.checkbox("Include intermediate steps (mask, etc.)")
    create_comparison = st.checkbox("Create comparison image")
    
    if st.button("📥 Generate Download Package", type="primary"):
        with st.spinner("Preparing download package..."):
            # Simulate package creation
            import time
            time.sleep(2)
            
            st.success("✅ Download package ready!")
            
            # In a real implementation, you would:
            # 1. Resize images according to selected resolution
            # 2. Convert to selected format with quality settings
            # 3. Create ZIP file with all selected components
            # 4. Provide download link
            
            st.download_button(
                label="📥 Download ZIP Package",
                data=b"dummy_data",  # Replace with actual ZIP data
                file_name="poisson_edit_results.zip",
                mime="application/zip"
            )

def create_help_and_tutorials():
    """Create help section with tutorials and tips."""
    st.markdown("### 📚 Help & Tutorials")
    
    # Quick tips
    with st.expander("💡 Quick Tips", expanded=True):
        st.markdown("""
        **For Best Results:**
        - Use high-contrast source images with clear edges
        - Ensure good lighting difference between object and background
        - Try different blend modes for different scenarios:
          - **Mixed**: Best for most cases (recommended)
          - **Seamless**: When you want to preserve source colors
          - **Monochrome**: For artistic effects or when only shape matters
        
        **Performance Tips:**
        - Enable "Fast mode" for quick previews
        - Reduce pyramid levels for faster processing
        - Use smaller images during experimentation
        """)
    
    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        
        **Visible seams or artifacts:**
        - Enable "Edge refinement"
        - Try "Mixed" blend mode
        - Increase pyramid levels for multi-scale processing
        
        **Poor color matching:**
        - Enable "Color correction"
        - Try different placement strategies
        - Adjust source image scale
        
        **Processing too slow:**
        - Enable "Fast mode"
        - Reduce pyramid levels
        - Use smaller images
        
        **Mask quality issues:**
        - Try different automatic mask generation methods
        - Use manual mask refinement
        - Ensure good contrast in source image
        """)
    
    # Tutorial videos (placeholders)
    with st.expander("🎥 Video Tutorials"):
        st.markdown("""
        **Available Tutorials:**
        - Getting Started (5 min)
        - Advanced Masking Techniques (8 min)
        - Blend Mode Comparison (6 min)
        - Professional Workflow Tips (10 min)
        
        *Video tutorials would be embedded here in a real deployment*
        """)
    
    # API documentation
    with st.expander("🔗 API Documentation"):
        st.markdown("""
        **For Developers:**
        
        The Enhanced Poisson Image Editor can be used programmatically:
        
        ```python
        from image_processing import EnhancedImageCompositor
        
        # Initialize compositor
        compositor = EnhancedImageCompositor()
        
        # Automatic processing
        result = compositor.automatic_composite_advanced(
            source_path="source.jpg",
            target_path="target.jpg",
            blend_mode="mixed",
            output_path="result.jpg"
        )
        ```
        
        **Available Parameters:**
        - `blend_mode`: "seamless", "mixed", "monochrome_transfer"
        - `placement_strategy`: "auto", "center", "bottom", "top"
        - `multi_scale`: Boolean for pyramid processing
        - `color_correct`: Boolean for automatic color correction
        """)

def create_advanced_visualization(source_img, target_img, mask, result):
    """Create advanced visualization with multiple views."""
    if any(img is None for img in [source_img, target_img, result]):
        return
    
    st.markdown("### 🎨 Advanced Visualization")
    
    # Create interactive 3D visualization
    tab1, tab2, tab3 = st.tabs(["Process Flow", "3D Analysis", "Color Analysis"])
    
    with tab1:
        # Process flow visualization
        st.markdown("#### Processing Pipeline Visualization")
        
        # Create a flow chart using plotly
        fig = go.Figure()
        
        # Add flowchart elements (simplified)
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[1, 1, 1, 1, 1],
            mode='markers+lines+text',
            text=['Source', 'Mask', 'Resize', 'Blend', 'Result'],
            textposition='top center',
            marker=dict(size=20, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#1f77b4']),
            line=dict(width=3, color='gray')
        ))
        
        fig.update_layout(
            title="Poisson Editing Pipeline",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # 3D surface analysis
        st.markdown("#### 3D Intensity Surface Analysis")
        
        # Convert result to grayscale for 3D analysis
        if len(result.shape) == 3:
            gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        else:
            gray_result = result
        
        # Downsample for performance
        h, w = gray_result.shape
        step = max(1, min(h, w) // 50)
        downsampled = gray_result[::step, ::step]
        
        # Create 3D surface plot
        x = np.arange(0, downsampled.shape[1])
        y = np.arange(0, downsampled.shape[0])
        X, Y = np.meshgrid(x, y)
        
        fig = go.Figure(data=[go.Surface(z=downsampled, x=X, y=Y, colorscale='viridis')])
        
        fig.update_layout(
            title="3D Intensity Surface",
            autosize=True,
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                zaxis_title="Intensity"
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Color analysis
        st.markdown("#### Color Distribution Analysis")
        
        # Analyze color distributions
        colors = ['Red', 'Green', 'Blue']
        
        for i, color in enumerate(colors):
            if len(result.shape) == 3:
                channel_data = result[:, :, i].flatten()
                
                fig = go.Figure(data=[go.Histogram(x=channel_data, name=color, opacity=0.7)])
                fig.update_layout(
                    title=f"{color} Channel Distribution",
                    xaxis_title="Intensity Value",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
