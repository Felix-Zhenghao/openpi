import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image, ImageDraw, ImageFont
import textwrap
import pathlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Configure matplotlib to avoid too many figures warning
plt.rcParams['figure.max_open_warning'] = 50  # Increase the warning threshold

class AttentionVisualizer:
    """Visualizes attention maps between action tokens and VLM KV cache tokens."""
    
    def __init__(self, output_dir="attention_viz", max_workers=None):
        """
        Initialize the attention visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.output_dir = output_dir
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create a custom colormap (white to red)
        self.cmap = LinearSegmentedColormap.from_list(
            "attention_heatmap", 
            [(1, 1, 1, 0), (1, 0, 0, 1)],
            N=256
        )
        
        # Set up thread pool for parallel processing
        self.max_workers = max_workers or min(32, multiprocessing.cpu_count() * 2)
    
    def visualize_attention(self, 
                           logits, 
                           images, 
                           prompt, 
                           inference_timestep=0,
                           visualize_last_diffusion_step_only=True,
                           visualize_last_layer_only=True):
        """
        Visualize attention maps for all diffusion timesteps, layers, and action tokens.
        
        Args:
            logits: Attention logits with shape [num_diffusion_timestep, num_transformer_layer, 
                                               bsz, num_kv_head, num_query_head, 51, 867]
            images: Dictionary of images used as input
            prompt: Text prompt used as input
            inference_timestep: Current inference timestep
            visualize_last_diffusion_step_only: If True, only visualize the last diffusion step
            visualize_last_layer_only: If True, only visualize the last transformer layer
        """
        # Create output directory for this inference timestep
        inference_dir = os.path.join(self.output_dir, f"inference_timestep_{inference_timestep}")
        pathlib.Path(inference_dir).mkdir(parents=True, exist_ok=True)
        
        # Get dimensions
        num_diffusion_timesteps, num_layers, _, _, num_query_heads, num_action_tokens, seq_len = logits.shape
        
        # Determine how many images we have (2 or 3)
        image_keys = list(images.keys())
        num_images = len(image_keys)
        
        # Calculate token ranges
        image_token_size = 256
        language_token_size = 48
        
        # Determine which diffusion steps and layers to visualize based on parameters
        diffusion_steps = [num_diffusion_timesteps - 1] if visualize_last_diffusion_step_only else range(num_diffusion_timesteps)
        layers = [num_layers - 1] if visualize_last_layer_only else range(num_layers)
        
        # Pre-compute all directory paths to avoid repeated string operations
        dir_structure = self._precompute_directory_structure(
            inference_dir, diffusion_steps, num_action_tokens, layers
        )
        
        # Create all directories at once
        all_dirs = [path for paths in dir_structure.values() for path in paths.values()]
        self._create_directories(all_dirs)
        
        # Prepare visualization tasks
        visualization_tasks = []
        
        # Process the entire logits tensor at once for better vectorization
        for diff_t in diffusion_steps:
            for action_idx in range(1, num_action_tokens):
                for layer_idx in layers:
                    # Get the directory for this configuration
                    layer_dir = dir_structure[diff_t][f"action_{action_idx-1}_layer_{layer_idx}"]
                    
                    # Get attention weights for this specific configuration
                    # Shape: [num_query_heads, seq_len]
                    attn_weights = logits[diff_t, layer_idx, 0, 0, :, action_idx, :]
                    
                    # Process images
                    for img_idx, img_key in enumerate(image_keys):
                        start_idx = img_idx * image_token_size
                        end_idx = start_idx + image_token_size
                        
                        # Get attention weights for this image across all heads
                        # Shape: [num_query_heads, image_token_size]
                        img_attn = attn_weights[:, start_idx:end_idx]
                        
                        # Create fused attention map (max across heads)
                        fused_attn = np.max(img_attn, axis=0)
                        
                        # Add image visualization task
                        output_path = os.path.join(layer_dir, f"fused_attention_on_{img_key}.jpg")
                        visualization_tasks.append(
                            (self._visualize_image_attention, (images[img_key], fused_attn, output_path))
                        )
                    
                    # Process text
                    text_start_idx = num_images * image_token_size
                    text_end_idx = text_start_idx + language_token_size
                    
                    # Get attention weights for text across all heads
                    # Shape: [num_query_heads, language_token_size]
                    text_attn = attn_weights[:, text_start_idx:text_end_idx]
                    
                    # Create fused attention map (max across heads)
                    fused_text_attn = np.max(text_attn, axis=0)
                    
                    # Add text visualization task
                    output_path = os.path.join(layer_dir, "fused_attention_on_text.jpg")
                    visualization_tasks.append(
                        (self._visualize_text_attention, (prompt, fused_text_attn, output_path))
                    )
        
        # Execute visualization tasks in parallel
        self._execute_tasks_parallel(visualization_tasks)
    
    def _precompute_directory_structure(self, inference_dir, diffusion_steps, num_action_tokens, layers):
        """
        Precompute all directory paths to avoid repeated string operations.
        
        Returns:
            Dictionary mapping diffusion timestep to a dictionary of action+layer paths
        """
        dir_structure = {}
        
        for diff_t in diffusion_steps:
            diff_dir = os.path.join(inference_dir, f"diffusion_timestep_{diff_t}")
            dir_structure[diff_t] = {}
            
            for action_idx in range(1, num_action_tokens):
                action_dir = os.path.join(diff_dir, f"action_token_{action_idx-1}")
                
                for layer_idx in layers:
                    layer_dir = os.path.join(action_dir, f"layer_{layer_idx}")
                    dir_structure[diff_t][f"action_{action_idx-1}_layer_{layer_idx}"] = layer_dir
        
        return dir_structure
    
    def _create_directories(self, directories):
        """Create all directories at once to minimize filesystem operations."""
        for directory in directories:
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _execute_tasks_parallel(self, tasks):
        """Execute visualization tasks in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for func, args in tasks:
                futures.append(executor.submit(func, *args))
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
    def _visualize_image_attention(self, image, attention_weights, output_path):
        """
        Visualize attention weights on an image.
        
        Args:
            image: Input image (H, W, 3)
            attention_weights: Attention weights for image tokens (256,)
            output_path: Path to save visualization
        """
        # Convert image to uint8 if it's float
        if image.dtype == np.float32:
            if image.min() < 0:  # [-1, 1] range
                image = (image + 1) * 127.5
            else:  # [0, 1] range
                image = image * 255
            image = image.astype(np.uint8)
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Reshape attention weights to a square (assuming 16x16 grid for 256 tokens)
        attn_map = attention_weights.reshape(16, 16)
        
        # Create a high-resolution attention map where each patch has uniform values
        # First, determine the size of each patch in the final image
        patch_h, patch_w = h // 16, w // 16
        
        # Create a high-resolution attention map
        high_res_attn = np.zeros((h, w), dtype=np.float32)
        
        # Fill each patch with the same attention value (no interpolation)
        for i in range(16):
            for j in range(16):
                y_start, y_end = i * patch_h, (i + 1) * patch_h
                x_start, x_end = j * patch_w, (j + 1) * patch_w
                high_res_attn[y_start:y_end, x_start:x_end] = attn_map[i, j]
        
        # Use the high-resolution attention map instead of resizing with interpolation
        attn_map = high_res_attn
        
        # Normalize attention weights to [0, 1] for visualization
        if attn_map.max() > 0:
            attn_map = attn_map / attn_map.max()
        
        # Convert image to BGR if it's grayscale
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_rgb = image.copy()
        
        # Create a visualization where only the red channel is affected by attention
        # Create a red mask where intensity corresponds to attention weight
        red_mask = np.zeros_like(image_rgb)
        red_mask[:, :, 2] = (attn_map * 255).astype(np.uint8)  # Red channel in BGR
        
        # Blend the original image with the red mask
        # This preserves the original image but adds red tint based on attention
        result = cv2.addWeighted(image_rgb, 1.0, red_mask, 0.5, 0)
        
        # Save the result
        cv2.imwrite(output_path, result)
    
    def _visualize_text_attention(self, text, attention_weights, output_path):
        """
        Visualize attention weights on text.
        
        Args:
            text: Input text prompt
            attention_weights: Attention weights for text tokens (48,)
            output_path: Path to save visualization
        """
        # Create a blank image
        width, height = 800, 200
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        # Wrap text to fit in image
        wrapped_text = textwrap.fill(text, width=80)
        
        # Normalize attention weights
        if attention_weights.max() > 0:
            norm_weights = attention_weights / attention_weights.max()
        else:
            norm_weights = attention_weights
        
        # Vectorized approach to extend weights to match text length
        char_weights = np.zeros(len(wrapped_text))
        
        # Create a mapping from character positions to token indices
        char_to_token = np.zeros(len(wrapped_text), dtype=int)
        token_idx = 0
        for i, char in enumerate(wrapped_text):
            char_to_token[i] = min(token_idx, len(norm_weights) - 1) if token_idx < len(norm_weights) else len(norm_weights) - 1
            if char == ' ':
                token_idx += 1
        
        # Vectorized assignment of weights
        valid_indices = char_to_token < len(norm_weights)
        char_weights[valid_indices] = norm_weights[char_to_token[valid_indices]]
        
        # Draw text with heatmap coloring
        x, y = 10, 10
        for i, char in enumerate(wrapped_text):
            # Get color based on attention weight
            weight = min(max(char_weights[i], 0), 1)
            r = int(255 * weight)
            g = int(255 * (1 - weight))
            b = int(255 * (1 - weight))
            color = (r, g, b)
            
            # Draw character
            draw.text((x, y), char, font=font, fill=color)
            
            # Get character width using getbbox() instead of getsize()
            # getbbox() returns (left, top, right, bottom)
            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            
            x += char_width
            
            # Move to next line if needed
            if char == '\n':
                x = 10
                y += font.getbbox('A')[3]  # Use height from getbbox
        
        # Save image
        img.save(output_path)

def visualize_attention_maps(logits, images, prompt, output_dir="attention_viz", inference_timestep=0, 
                            visualize_last_diffusion_step_only=True, visualize_last_layer_only=True):
    """
    Wrapper function to visualize attention maps.
    
    Args:
        logits: Attention logits with shape [num_diffusion_timestep, num_transformer_layer, 
                                           bsz, num_kv_head, num_query_head, 51, 867]
        images: Dictionary of images used as input
        prompt: Text prompt used as input
        output_dir: Directory to save visualizations
        inference_timestep: Current inference timestep
        visualize_last_diffusion_step_only: If True, only visualize the last diffusion step
        visualize_last_layer_only: If True, only visualize the last transformer layer
    """
    visualizer = AttentionVisualizer(output_dir)
    visualizer.visualize_attention(
        logits, 
        images, 
        prompt, 
        inference_timestep,
        visualize_last_diffusion_step_only,
        visualize_last_layer_only
    )