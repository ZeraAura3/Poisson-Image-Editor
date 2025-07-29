import numpy as np
import cv2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import os
from skimage import segmentation, filters, morphology, measure
from skimage.feature import canny
from skimage.color import rgb2lab, lab2rgb, rgb2xyz, xyz2rgb
import warnings
warnings.filterwarnings('ignore')


class AdvancedImageBlender:
    """
    Advanced Poisson Image Blender with multi-scale processing, gradient mixing,
    and enhanced boundary handling for superior blending results.
    """
    
    def __init__(self, num_levels=4, min_size=32):
        """
        Initialize the Advanced Image Blender.
        
        Args:
            num_levels (int): Number of pyramid levels for multi-scale blending
            min_size (int): Minimum image size for pyramid levels
        """
        self.num_levels = num_levels
        self.min_size = min_size
    
    @staticmethod
    def create_advanced_mask(img, interactive=False, use_grabcut=True, refine_edges=True):
        """
        Create an advanced mask with multiple techniques for better object extraction.
        
        Args:
            img: Input image
            interactive: Whether to use interactive drawing
            use_grabcut: Whether to use GrabCut algorithm
            refine_edges: Whether to refine mask edges
        """
        if interactive:
            # For web interface, we'll return an automatic mask
            # Interactive functionality would require a different implementation
            return AdvancedImageBlender._create_automatic_mask(img, use_grabcut, refine_edges)
        else:
            # Advanced automatic mask creation
            return AdvancedImageBlender._create_automatic_mask(img, use_grabcut, refine_edges)
    
    @staticmethod
    def _create_automatic_mask(img, use_grabcut=True, refine_edges=True):
        """Create mask automatically using multiple computer vision techniques."""
        # Convert to different color spaces for better segmentation
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
        
        # Method 2: Edge-based segmentation
        edges = canny(gray, sigma=1.0, low_threshold=50, high_threshold=150)
        edges = morphology.binary_closing(edges, morphology.disk(2))
        
        # Method 3: Color-based segmentation using K-means
        h, w = img.shape[:2]
        img_flat = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(img_flat, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Find the cluster with pixels near the border (likely background)
        border_pixels = np.concatenate([
            img_flat[:w], img_flat[-w:],  # top and bottom rows
            img_flat[::w], img_flat[w-1::w]  # left and right columns
        ])
        border_labels = np.concatenate([
            labels[:w], labels[-w:],
            labels[::w], labels[w-1::w]
        ])
        
        # Most common label in border is likely background
        bg_label = np.bincount(border_labels.flatten()).argmax()
        kmeans_mask = (labels != bg_label).reshape(h, w).astype(np.uint8)
        
        # Combine methods
        combined_mask = np.logical_or(
            np.logical_or(adaptive_thresh > 0, edges),
            kmeans_mask
        ).astype(np.uint8)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Keep only the largest connected component
        num_labels, labels_im = cv2.connectedComponents(combined_mask)
        if num_labels > 1:
            # Find largest component (excluding background)
            sizes = [np.sum(labels_im == i) for i in range(1, num_labels)]
            largest_component = np.argmax(sizes) + 1
            combined_mask = (labels_im == largest_component).astype(np.uint8)
        
        # Apply GrabCut refinement if requested
        if use_grabcut and np.sum(combined_mask) > 0.01 * combined_mask.size:
            combined_mask = AdvancedImageBlender._refine_with_grabcut(img, combined_mask)
        
        # Apply edge refinement if requested
        if refine_edges:
            combined_mask = AdvancedImageBlender._refine_mask_edges(img, combined_mask)
        
        return combined_mask
    
    @staticmethod
    def _refine_with_grabcut(img, initial_mask):
        """Refine mask using GrabCut algorithm."""
        try:
            # Create GrabCut mask
            mask_grabcut = np.zeros(img.shape[:2], np.uint8)
            mask_grabcut[initial_mask == 1] = cv2.GC_FGD  # Foreground
            mask_grabcut[initial_mask == 0] = cv2.GC_BGD  # Background
            
            # Set probable foreground/background around the boundary
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            dilated = cv2.dilate(initial_mask, kernel, iterations=1)
            eroded = cv2.erode(initial_mask, kernel, iterations=1)
            
            mask_grabcut[np.logical_and(dilated == 1, initial_mask == 0)] = cv2.GC_PR_BGD
            mask_grabcut[np.logical_and(initial_mask == 1, eroded == 0)] = cv2.GC_PR_FGD
            
            # Apply GrabCut
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img, mask_grabcut, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            
            # Extract final mask
            refined_mask = np.where((mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            return refined_mask
        except:
            return initial_mask
    
    @staticmethod
    def _refine_mask_edges(img, mask):
        """Refine mask edges using image gradients."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
        
        # Apply bilateral filter for edge-preserving smoothing
        mask_float = mask.astype(np.float32)
        refined_mask = cv2.bilateralFilter(mask_float, 9, 75, 75)
        
        # Use gradient information to refine boundaries
        boundary = cv2.dilate(mask, np.ones((3,3), np.uint8)) - cv2.erode(mask, np.ones((3,3), np.uint8))
        boundary_indices = np.where(boundary > 0)
        
        # Adjust mask values at boundaries based on gradient strength
        for i, j in zip(boundary_indices[0], boundary_indices[1]):
            if gradient_magnitude[i, j] > 50:  # Strong edge
                # Keep original mask value
                refined_mask[i, j] = mask[i, j]
            else:
                # Apply more smoothing
                refined_mask[i, j] = refined_mask[i, j] * 0.8 + mask[i, j] * 0.2
        
        return (refined_mask > 0.5).astype(np.uint8)
    
    @staticmethod
    def get_enhanced_laplacian_matrix(height, width, boundary_handling='dirichlet'):
        """
        Create an enhanced Laplacian matrix with better boundary handling.
        
        Args:
            height, width: Image dimensions
            boundary_handling: 'dirichlet', 'neumann', or 'mixed'
        """
        n_pixels = height * width
        row_indices = []
        col_indices = []
        data = []
        
        def get_index(i, j):
            return i * width + j
        
        # For each pixel
        for i in range(height):
            for j in range(width):
                current_idx = get_index(i, j)
                
                # Handle boundary conditions
                if boundary_handling == 'neumann' and (i == 0 or i == height-1 or j == 0 or j == width-1):
                    # Neumann boundary: derivative = 0
                    neighbors = []
                    coefficients = []
                    
                    # Add valid neighbors
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbors.append(get_index(ni, nj))
                            coefficients.append(-1)
                    
                    # Center coefficient
                    center_coeff = len(neighbors)
                    
                    # Add to sparse matrix
                    row_indices.append(current_idx)
                    col_indices.append(current_idx)
                    data.append(center_coeff)
                    
                    for neighbor, coeff in zip(neighbors, coefficients):
                        row_indices.append(current_idx)
                        col_indices.append(neighbor)
                        data.append(coeff)
                else:
                    # Standard Laplacian (Dirichlet boundary)
                    row_indices.append(current_idx)
                    col_indices.append(current_idx)
                    data.append(4)
                    
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_idx = get_index(ni, nj)
                            row_indices.append(current_idx)
                            col_indices.append(neighbor_idx)
                            data.append(-1)
        
        return sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_pixels, n_pixels))
    
    def create_image_pyramid(self, img):
        """Create Gaussian pyramid for multi-scale processing."""
        pyramid = [img.astype(np.float64)]
        current_img = img.astype(np.float64)
        
        for level in range(self.num_levels - 1):
            height, width = current_img.shape[:2]
            if height <= self.min_size or width <= self.min_size:
                break
            
            # Apply Gaussian blur before downsampling
            if len(current_img.shape) == 3:
                blurred = np.zeros_like(current_img)
                for c in range(current_img.shape[2]):
                    blurred[:, :, c] = gaussian_filter(current_img[:, :, c], sigma=1.0)
            else:
                blurred = gaussian_filter(current_img, sigma=1.0)
            
            # Downsample
            current_img = cv2.pyrDown(blurred.astype(np.uint8)).astype(np.float64)
            pyramid.append(current_img)
        
        return pyramid
    
    def advanced_poisson_blend(self, source_img, target_img, mask, offset=(0, 0), 
                              blend_mode='seamless', color_correct=True, multi_scale=True):
        """
        Advanced Poisson blending with multiple enhancement techniques.
        
        Args:
            source_img, target_img: Input images
            mask: Binary mask
            offset: Position offset
            blend_mode: 'seamless', 'mixed', 'monochrome_transfer'
            color_correct: Whether to apply color correction
            multi_scale: Whether to use multi-scale blending
        """
        # Prepare images
        source_img = source_img.astype(np.float64)
        target_img = target_img.astype(np.float64)
        
        if color_correct:
            source_img = self._apply_color_correction(source_img, target_img, mask, offset)
        
        if multi_scale and blend_mode != 'monochrome_transfer':
            return self._multi_scale_blend(source_img, target_img, mask, offset, blend_mode)
        else:
            return self._single_scale_blend(source_img, target_img, mask, offset, blend_mode)
    
    def _apply_color_correction(self, source_img, target_img, mask, offset):
        """Apply color correction to match source and target color characteristics."""
        # Convert to LAB color space for better color handling
        source_lab = cv2.cvtColor(source_img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float64)
        target_lab = cv2.cvtColor(target_img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float64)
        
        # Create offset mask
        h, w = target_img.shape[:2]
        offset_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Calculate placement region
        y_min, x_min = offset
        mask_h, mask_w = mask.shape[:2]
        y_max = min(y_min + mask_h, h)
        x_max = min(x_min + mask_w, w)
        
        # Adjust for negative offsets
        src_y_start = abs(min(0, y_min))
        src_x_start = abs(min(0, x_min))
        target_y_start = max(0, y_min)
        target_x_start = max(0, x_min)
        
        src_y_end = src_y_start + (y_max - target_y_start)
        src_x_end = src_x_start + (x_max - target_x_start)
        
        if src_y_start < src_y_end and src_x_start < src_x_end:
            offset_mask[target_y_start:y_max, target_x_start:x_max] = \
                mask[src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Get boundary region for color matching
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        boundary_region = cv2.dilate(offset_mask, kernel) - offset_mask
        
        if np.sum(boundary_region) > 0:
            # Calculate color statistics for boundary region
            target_boundary = target_lab[boundary_region > 0]
            
            # Map source to target boundary
            source_boundary_indices = []
            for i in range(h):
                for j in range(w):
                    if boundary_region[i, j] > 0:
                        # Find corresponding source position
                        src_i = i - target_y_start + src_y_start
                        src_j = j - target_x_start + src_x_start
                        if (0 <= src_i < source_lab.shape[0] and 
                            0 <= src_j < source_lab.shape[1]):
                            source_boundary_indices.append((src_i, src_j))
            
            if source_boundary_indices:
                # Apply color transfer using LAB statistics
                source_corrected = source_img.copy()
                source_lab_corrected = source_lab.copy()
                
                target_mean = np.mean(target_boundary, axis=0)
                target_std = np.std(target_boundary, axis=0)
                
                source_boundary_values = np.array([source_lab[i, j] for i, j in source_boundary_indices])
                if len(source_boundary_values) > 0:
                    source_mean = np.mean(source_boundary_values, axis=0)
                    source_std = np.std(source_boundary_values, axis=0)
                    
                    # Apply color transfer
                    for c in range(3):
                        if source_std[c] > 0:
                            source_lab_corrected[:, :, c] = (
                                (source_lab_corrected[:, :, c] - source_mean[c]) * 
                                (target_std[c] / source_std[c]) + target_mean[c]
                            )
                
                # Convert back to BGR
                source_lab_corrected = np.clip(source_lab_corrected, 0, 255)
                source_corrected = cv2.cvtColor(source_lab_corrected.astype(np.uint8), cv2.COLOR_LAB2BGR)
                return source_corrected.astype(np.float64)
        
        return source_img
    
    def _multi_scale_blend(self, source_img, target_img, mask, offset, blend_mode):
        """Perform multi-scale Poisson blending."""
        # Create pyramids
        source_pyramid = self.create_image_pyramid(source_img)
        target_pyramid = self.create_image_pyramid(target_img)
        
        # Create mask pyramid
        mask_pyramid = []
        current_mask = mask.astype(np.float64)
        for level in range(len(source_pyramid)):
            if level == 0:
                mask_pyramid.append(mask)
            else:
                current_mask = cv2.pyrDown(current_mask.astype(np.uint8)).astype(np.float64)
                mask_pyramid.append((current_mask > 0.5).astype(np.uint8))
        
        # Blend at each scale, starting from coarsest
        result_pyramid = []
        scale_factor = 2 ** (len(source_pyramid) - 1)
        
        for level in range(len(source_pyramid) - 1, -1, -1):
            current_source = source_pyramid[level]
            current_target = target_pyramid[level]
            current_mask = mask_pyramid[level]
            
            # Scale offset for current level
            level_scale = 2 ** level
            scaled_offset = (offset[0] // level_scale, offset[1] // level_scale)
            
            if level == len(source_pyramid) - 1:
                # Coarsest level - regular blend
                result = self._single_scale_blend(current_source, current_target, 
                                                current_mask, scaled_offset, blend_mode)
            else:
                # Finer levels - use previous result as initialization
                prev_result = cv2.pyrUp(result_pyramid[0].astype(np.uint8)).astype(np.float64)
                
                # Adjust size if needed
                if prev_result.shape[:2] != current_target.shape[:2]:
                    prev_result = cv2.resize(prev_result, 
                                           (current_target.shape[1], current_target.shape[0]))
                
                # Blend with detail preservation
                result = self._detail_preserving_blend(current_source, current_target, 
                                                     current_mask, scaled_offset, 
                                                     prev_result, blend_mode)
            
            result_pyramid.insert(0, result)
        
        return result_pyramid[0].astype(np.uint8)
    
    def _single_scale_blend(self, source_img, target_img, mask, offset, blend_mode):
        """Perform single-scale Poisson blending with different modes."""
        h, w = target_img.shape[:2]
        channels = target_img.shape[2] if len(target_img.shape) > 2 else 1
        
        # Create offset mask
        offset_mask = self._create_offset_mask(mask, target_img.shape[:2], offset)
        
        # Map source to target domain
        source_mapped = self._map_source_to_target(source_img, target_img.shape[:2], mask, offset)
        
        result = target_img.copy()
        
        # Get enhanced Laplacian matrix
        L = self.get_enhanced_laplacian_matrix(h, w, 'mixed')
        
        # Process each channel
        for c in range(channels):
            source_channel = source_mapped[:, :, c] if channels > 1 else source_mapped
            target_channel = target_img[:, :, c] if channels > 1 else target_img
            
            # Calculate gradients based on blend mode
            if blend_mode == 'seamless':
                guidance_field = self._calculate_guidance_field(source_channel, target_channel, 
                                                               offset_mask, 'source')
            elif blend_mode == 'mixed':
                guidance_field = self._calculate_guidance_field(source_channel, target_channel, 
                                                               offset_mask, 'mixed')
            elif blend_mode == 'monochrome_transfer':
                if c == 0:  # Only process first channel for monochrome
                    guidance_field = self._calculate_guidance_field(source_channel, target_channel, 
                                                                   offset_mask, 'source')
                else:
                    result[:, :, c] = target_channel  # Keep target colors
                    continue
            
            # Solve Poisson equation
            blended_channel = self._solve_poisson_equation(L, guidance_field, target_channel, offset_mask)
            
            if channels > 1:
                result[:, :, c] = blended_channel
            else:
                result = blended_channel
        
        return result
    
    def _detail_preserving_blend(self, source_img, target_img, mask, offset, prev_result, blend_mode):
        """Blend with detail preservation from multi-scale processing."""
        # Calculate detail layer (high-frequency components)
        target_smooth = gaussian_filter(target_img, sigma=2.0) if len(target_img.shape) == 2 else \
                       np.stack([gaussian_filter(target_img[:, :, c], sigma=2.0) for c in range(target_img.shape[2])], axis=2)
        
        target_detail = target_img - target_smooth
        
        # Regular blend
        base_result = self._single_scale_blend(source_img, target_smooth, mask, offset, blend_mode)
        
        # Add back details outside the mask region
        offset_mask = self._create_offset_mask(mask, target_img.shape[:2], offset)
        detail_weight = 1.0 - gaussian_filter(offset_mask.astype(np.float64), sigma=5.0)
        
        if len(target_img.shape) == 3:
            detail_weight = np.stack([detail_weight] * target_img.shape[2], axis=2)
        
        result = base_result + target_detail * detail_weight
        
        return np.clip(result, 0, 255)
    
    def _create_offset_mask(self, mask, target_shape, offset):
        """Create mask with proper offset in target image coordinate system."""
        h, w = target_shape
        offset_mask = np.zeros((h, w), dtype=np.uint8)
        
        y_min, x_min = offset
        mask_h, mask_w = mask.shape[:2]
        
        y_max = min(y_min + mask_h, h)
        x_max = min(x_min + mask_w, w)
        
        src_y_start = abs(min(0, y_min))
        src_x_start = abs(min(0, x_min))
        target_y_start = max(0, y_min)
        target_x_start = max(0, x_min)
        
        src_y_end = src_y_start + (y_max - target_y_start)
        src_x_end = src_x_start + (x_max - target_x_start)
        
        if src_y_start < src_y_end and src_x_start < src_x_end:
            offset_mask[target_y_start:y_max, target_x_start:x_max] = \
                mask[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return offset_mask
    
    def _map_source_to_target(self, source_img, target_shape, mask, offset):
        """Map source image to target coordinate system."""
        h, w = target_shape
        channels = source_img.shape[2] if len(source_img.shape) > 2 else 1
        
        if channels > 1:
            source_mapped = np.zeros((h, w, channels))
        else:
            source_mapped = np.zeros((h, w))
        
        y_min, x_min = offset
        mask_h, mask_w = mask.shape[:2]
        
        y_max = min(y_min + mask_h, h)
        x_max = min(x_min + mask_w, w)
        
        src_y_start = abs(min(0, y_min))
        src_x_start = abs(min(0, x_min))
        target_y_start = max(0, y_min)
        target_x_start = max(0, x_min)
        
        src_y_end = src_y_start + (y_max - target_y_start)
        src_x_end = src_x_start + (x_max - target_x_start)
        
        if src_y_start < src_y_end and src_x_start < src_x_end:
            if channels > 1:
                source_mapped[target_y_start:y_max, target_x_start:x_max] = \
                    source_img[src_y_start:src_y_end, src_x_start:src_x_end]
            else:
                source_mapped[target_y_start:y_max, target_x_start:x_max] = \
                    source_img[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return source_mapped
    
    def _calculate_guidance_field(self, source_channel, target_channel, mask, mode):
        """Calculate guidance field for Poisson equation."""
        h, w = source_channel.shape
        guidance = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 1:
                    # Calculate Laplacian
                    center_src = source_channel[i, j]
                    center_tgt = target_channel[i, j]
                    
                    laplacian_src = 0
                    laplacian_tgt = 0
                    
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            laplacian_src += center_src - source_channel[ni, nj]
                            laplacian_tgt += center_tgt - target_channel[ni, nj]
                    
                    if mode == 'source':
                        guidance[i, j] = laplacian_src
                    elif mode == 'mixed':
                        # Choose gradient with larger magnitude
                        if abs(laplacian_src) > abs(laplacian_tgt):
                            guidance[i, j] = laplacian_src
                        else:
                            guidance[i, j] = laplacian_tgt
        
        return guidance
    
    def _solve_poisson_equation(self, L, guidance_field, target_channel, mask):
        """Solve the Poisson equation with given guidance field."""
        h, w = target_channel.shape
        
        # Flatten arrays
        mask_flat = mask.flatten()
        target_flat = target_channel.flatten()
        guidance_flat = guidance_field.flatten()
        
        # Find interior and boundary indices
        interior_indices = np.where(mask_flat == 1)[0]
        boundary_indices = np.where(mask_flat == 0)[0]
        
        if len(interior_indices) == 0:
            return target_channel
        
        # Create system matrix
        n_pixels = h * w
        interior_mask = np.zeros(n_pixels)
        interior_mask[interior_indices] = 1
        interior_diag = sparse.diags(interior_mask)
        
        # Boundary conditions
        boundary_data = np.ones(len(boundary_indices))
        boundary_matrix = sparse.csr_matrix((boundary_data, (boundary_indices, boundary_indices)), 
                                          shape=(n_pixels, n_pixels))
        
        # Combined system
        A = interior_diag.dot(L) + boundary_matrix
        
        # Right-hand side
        b = np.zeros(n_pixels)
        b[interior_indices] = guidance_flat[interior_indices]
        b[boundary_indices] = target_flat[boundary_indices]
        
        # Solve system
        x = spsolve(A, b)
        result = x.reshape((h, w))
        
        return np.clip(result, 0, 255)


class EnhancedImageCompositor:
    """
    Enhanced Image Compositor with advanced Poisson blending, multi-scale processing,
    and intelligent automatic placement.
    """
    
    def __init__(self, num_levels=4, min_size=32):
        """Initialize the Enhanced Image Compositor."""
        self.blender = AdvancedImageBlender(num_levels, min_size)
        self.num_levels = num_levels
        self.min_size = min_size
    
    @staticmethod
    def analyze_image_content(img):
        """Analyze image content to determine optimal placement strategies."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        
        # Calculate image statistics
        stats = {
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'edge_density': np.mean(canny(gray, sigma=1.0)),
            'contrast': np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        }
        
        # Simple saliency calculation (center bias)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        stats['saliency_center'] = (center_y, center_x)
        
        return stats
    
    def smart_resize_and_placement(self, source_img, target_img, placement_strategy='auto'):
        """
        Intelligently resize and place source image based on content analysis.
        
        Args:
            source_img, target_img: Input images
            placement_strategy: 'auto', 'center', 'bottom', 'saliency_based'
        """
        source_stats = self.analyze_image_content(source_img)
        target_stats = self.analyze_image_content(target_img)
        
        # Determine optimal scale based on content
        source_area = source_img.shape[0] * source_img.shape[1]
        target_area = target_img.shape[0] * target_img.shape[1]
        area_ratio = source_area / target_area
        
        # Adaptive scaling based on image characteristics
        if area_ratio > 0.5:  # Large source
            base_scale = 0.2
        elif area_ratio > 0.1:  # Medium source
            base_scale = 0.4
        else:  # Small source
            base_scale = 0.6
        
        # Adjust based on contrast and edge density
        contrast_factor = min(2.0, max(0.5, source_stats['contrast'] / target_stats['contrast']))
        scale_factor = base_scale * contrast_factor
        scale_factor = np.clip(scale_factor, 0.1, 0.8)
        
        # Resize source
        new_height = int(source_img.shape[0] * scale_factor)
        new_width = int(source_img.shape[1] * scale_factor)
        source_resized = cv2.resize(source_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Determine placement
        if placement_strategy == 'auto':
            if target_stats['saliency_center'][0] < target_img.shape[0] * 0.3:
                # Salient content at top, place at bottom
                placement_strategy = 'bottom'
            elif target_stats['saliency_center'][0] > target_img.shape[0] * 0.7:
                # Salient content at bottom, place at top
                placement_strategy = 'top'
            else:
                placement_strategy = 'saliency_based'
        
        # Calculate offset based on strategy
        if placement_strategy == 'center':
            offset_y = (target_img.shape[0] - new_height) // 2
            offset_x = (target_img.shape[1] - new_width) // 2
        elif placement_strategy == 'bottom':
            offset_y = target_img.shape[0] - new_height - 50
            offset_x = (target_img.shape[1] - new_width) // 2
        elif placement_strategy == 'top':
            offset_y = 50
            offset_x = (target_img.shape[1] - new_width) // 2
        elif placement_strategy == 'saliency_based':
            # Place away from salient regions
            saliency_y, saliency_x = target_stats['saliency_center']
            if saliency_y < target_img.shape[0] // 2:
                offset_y = target_img.shape[0] - new_height - 50
            else:
                offset_y = 50
            offset_x = (target_img.shape[1] - new_width) // 2
        
        # Ensure offsets are within bounds
        offset_y = max(0, min(offset_y, target_img.shape[0] - new_height))
        offset_x = max(0, min(offset_x, target_img.shape[1] - new_width))
        
        return source_resized, (offset_y, offset_x), scale_factor
