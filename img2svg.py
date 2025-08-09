#!/usr/bin/env python3
"""
Image to Printable SVG Converter
Converts raster images to SVGs with limited color palettes suitable for 3D printing.
"""

import argparse
import sys
import os
import re
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter
# potrace is called via subprocess instead of Python bindings
import tempfile
import subprocess
import xml.etree.ElementTree as ET


class ImageToSVGConverter:
    """Main converter class for image to printable SVG conversion."""
    
    def __init__(self, n_colors=8, method='kmeans', simplify=True, threshold=128, include_background=False, suggested_colors=None, denoise=False, denoise_strength=3):
        self.n_colors = n_colors
        self.method = method
        self.simplify = simplify
        self.threshold = threshold
        self.include_background = include_background
        self.suggested_colors = suggested_colors  # List of (R, G, B) tuples
        self.palette = None
        self.unlimited_colors = (n_colors == 0)
        self.denoise = denoise
        self.denoise_strength = denoise_strength
    
    def convert(self, input_path, output_path):
        """
        Convert an image to a printable SVG with limited colors.
        
        Args:
            input_path: Path to input image file
            output_path: Path to output SVG file
        """
        print(f"Converting {input_path} to printable SVG...")
        
        # Load and process image
        image = Image.open(input_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply denoising if requested
        if self.denoise:
            print(f"Applying denoising (strength: {self.denoise_strength})...")
            image = self.denoise_image(image)
        
        # Reduce colors (unless unlimited)
        if self.unlimited_colors:
            print(f"Processing with unlimited colors...")
            # For unlimited colors, apply smart color reduction to merge very similar colors
            # This prevents having tens of thousands of nearly identical colors
            quantized_image = self.smart_color_reduction(image)
        else:
            print(f"Reducing to {self.n_colors} colors using {self.method} method...")
            quantized_image = self.quantize_image(image)
        
        # Convert to SVG
        print("Converting to SVG...")
        svg_content = self.image_to_svg(quantized_image)
        
        # Save SVG
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        print(f"SVG saved to {output_path}")
        return output_path
    
    def denoise_image(self, image):
        """Apply denoising to reduce artifacts and noise in the image."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply median filter to each channel
        if self.denoise_strength > 0:
            # Use scipy's median filter for better control
            for i in range(3):  # RGB channels
                img_array[:, :, i] = median_filter(img_array[:, :, i], size=self.denoise_strength)
        
        # Also apply a slight Gaussian blur to smooth out remaining noise
        denoised = Image.fromarray(img_array)
        if self.denoise_strength > 1:
            denoised = denoised.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return denoised
    
    def smart_color_reduction(self, image, tolerance=8):
        """Reduce colors by merging very similar ones. Used for unlimited colors mode.
        
        Args:
            image: PIL Image
            tolerance: Color difference threshold for merging (0-255)
        """
        # Convert to numpy array
        img_array = np.array(image)
        original_shape = img_array.shape
        pixels = img_array.reshape(-1, 3)
        
        # Get unique colors and their counts
        unique_colors, inverse_indices, counts = np.unique(
            pixels, axis=0, return_inverse=True, return_counts=True
        )
        
        print(f"  Found {len(unique_colors)} unique colors")
        
        # If we have a reasonable number of colors, just return as-is
        if len(unique_colors) <= 1000:
            print(f"  Using all {len(unique_colors)} colors")
            return image
        
        # Otherwise, merge similar colors
        print(f"  Merging similar colors (tolerance: {tolerance})...")
        
        # Sort colors by frequency (most common first)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_colors = unique_colors[sorted_indices]
        sorted_counts = counts[sorted_indices]
        
        # Build merged color palette
        merged_colors = []
        color_mapping = {}
        
        for i, color in enumerate(sorted_colors):
            # Check if this color is similar to any in merged_colors
            merged = False
            for j, merged_color in enumerate(merged_colors):
                # Calculate color distance
                diff = np.abs(color.astype(int) - merged_color.astype(int))
                if np.max(diff) <= tolerance:
                    # Map this color to the existing merged color
                    color_mapping[tuple(color)] = j
                    merged = True
                    break
            
            if not merged:
                # Add as new color
                color_mapping[tuple(color)] = len(merged_colors)
                merged_colors.append(color)
        
        merged_colors = np.array(merged_colors)
        print(f"  Reduced to {len(merged_colors)} colors")
        
        # Map pixels to merged colors
        new_pixels = np.zeros_like(pixels)
        for i, color in enumerate(unique_colors):
            mask = inverse_indices == i
            merged_idx = color_mapping[tuple(color)]
            new_pixels[mask] = merged_colors[merged_idx]
        
        # Reshape and return
        new_array = new_pixels.reshape(original_shape)
        return Image.fromarray(new_array.astype(np.uint8))
    
    def quantize_image(self, image):
        """Reduce the number of colors in an image."""
        # Convert to numpy array
        img_array = np.array(image)
        original_shape = img_array.shape
        
        # Reshape for clustering
        pixels = img_array.reshape(-1, 3)
        
        if self.method == 'kmeans':
            # Handle single color (silhouette)
            if self.n_colors == 1:
                # For silhouette, use the average color
                mean_color = np.mean(pixels, axis=0).astype(int)
                self.palette = np.array([mean_color])
                quantized_pixels = np.full_like(pixels, mean_color)
            # Use k-means clustering
            elif self.suggested_colors and len(self.suggested_colors) > 0:
                # If we have suggested colors, use them as initial centers
                n_suggested = min(len(self.suggested_colors), self.n_colors)
                n_remaining = self.n_colors - n_suggested
                
                if n_remaining > 0:
                    # Use k-means to find additional colors
                    kmeans = KMeans(n_clusters=n_remaining, random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    
                    # Combine suggested colors with k-means centers
                    suggested_array = np.array(self.suggested_colors[:n_suggested])
                    kmeans_centers = kmeans.cluster_centers_.astype(int)
                    self.palette = np.vstack([suggested_array, kmeans_centers])
                else:
                    # Use only suggested colors
                    self.palette = np.array(self.suggested_colors[:self.n_colors])
                
                # Now cluster all pixels using the combined palette
                from scipy.spatial.distance import cdist
                distances = cdist(pixels, self.palette)
                labels = np.argmin(distances, axis=1)
                quantized_pixels = self.palette[labels]
            else:
                # Standard k-means without suggested colors
                kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Get the color palette
                self.palette = kmeans.cluster_centers_.astype(int)
                
                # Replace pixels with cluster centers
                labels = kmeans.predict(pixels)
                quantized_pixels = self.palette[labels]
            
        elif self.method == 'posterize':
            # Simple posterization
            # For 1 color, just make everything the same
            if self.n_colors == 1:
                mean_color = np.mean(pixels, axis=0).astype(int)
                quantized_pixels = np.full_like(pixels, mean_color)
                self.palette = np.array([mean_color])
            else:
                levels = max(2, int((256 / self.n_colors) ** (1/3)))
                factor = 256 // levels
                quantized_pixels = (pixels // factor) * factor
                
                # Extract unique colors as palette
                unique_colors = np.unique(quantized_pixels, axis=0)
                if len(unique_colors) > self.n_colors:
                    # Use k-means to reduce further
                    kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
                    kmeans.fit(unique_colors)
                    self.palette = kmeans.cluster_centers_.astype(int)
                    
                    # Map colors to palette
                    labels = kmeans.predict(quantized_pixels)
                    quantized_pixels = self.palette[labels]
                else:
                    self.palette = unique_colors
        
        elif self.method == 'adaptive':
            # Adaptive palette using PIL's quantize
            if self.suggested_colors and len(self.suggested_colors) > 0:
                # If we have suggested colors, combine with adaptive palette
                n_suggested = min(len(self.suggested_colors), self.n_colors)
                n_remaining = self.n_colors - n_suggested
                
                if n_remaining > 0:
                    # Get additional colors using adaptive quantization
                    pil_image = Image.fromarray(img_array)
                    quantized = pil_image.quantize(colors=n_remaining, method=Image.Quantize.MEDIANCUT)
                    
                    # Get the palette from the quantized image
                    palette = quantized.getpalette()
                    if palette:
                        adaptive_colors = []
                        for i in range(0, min(n_remaining * 3, len(palette)), 3):
                            if i + 2 < len(palette):
                                adaptive_colors.append([palette[i], palette[i+1], palette[i+2]])
                        
                        # Combine suggested and adaptive colors
                        suggested_array = np.array(self.suggested_colors[:n_suggested])
                        adaptive_array = np.array(adaptive_colors)
                        self.palette = np.vstack([suggested_array, adaptive_array])
                    else:
                        # Just use suggested colors
                        self.palette = np.array(self.suggested_colors[:self.n_colors])
                else:
                    # Use only suggested colors
                    self.palette = np.array(self.suggested_colors[:self.n_colors])
                
                # Map pixels to nearest palette color
                from scipy.spatial.distance import cdist
                distances = cdist(pixels, self.palette)
                labels = np.argmin(distances, axis=1)
                quantized_pixels = self.palette[labels]
            else:
                # Standard adaptive quantization
                pil_image = Image.fromarray(img_array)
                quantized = pil_image.quantize(colors=self.n_colors, method=Image.Quantize.MEDIANCUT)
                
                # Get the palette from the quantized image
                palette = quantized.getpalette()
                if palette:
                    # Extract only the colors we need (palette might be longer)
                    # PIL returns palette as a flat list [r1,g1,b1,r2,g2,b2,...]
                    self.palette = []
                    for i in range(0, min(self.n_colors * 3, len(palette)), 3):
                        if i + 2 < len(palette):
                            self.palette.append([palette[i], palette[i+1], palette[i+2]])
                    self.palette = np.array(self.palette)
                else:
                    # Fallback if no palette found
                    print("Warning: No palette found in adaptive quantization, falling back to unique colors")
                    quantized_rgb = quantized.convert('RGB')
                    unique_colors = np.unique(np.array(quantized_rgb).reshape(-1, 3), axis=0)
                    self.palette = unique_colors[:self.n_colors]
                
                # Convert back to RGB and get pixels
                quantized_rgb = quantized.convert('RGB')
                quantized_pixels = np.array(quantized_rgb).reshape(-1, 3)
        
        # Reshape back to image
        quantized_array = quantized_pixels.reshape(original_shape)
        return Image.fromarray(quantized_array.astype(np.uint8))
    
    def image_to_svg(self, image):
        """Convert a quantized image to SVG using potrace for each color layer."""
        width, height = image.size
        svg_parts = []
        
        # SVG header - use pixels explicitly
        svg_parts.append(f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}px" height="{height}px" 
     viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg">''')
        
        # Process each color separately
        img_array = np.array(image)
        unique_colors = np.unique(img_array.reshape(-1, 3), axis=0)
        
        print(f"Processing {len(unique_colors)} unique colors...")
        
        # Only show detailed progress for reasonable number of colors
        show_progress = len(unique_colors) <= 100
        
        for i, color in enumerate(unique_colors):
            if show_progress:
                print(f"  Processing color {i+1}/{len(unique_colors)}: RGB{tuple(color)}")
            elif i % 1000 == 0:  # Show progress every 1000 colors for unlimited mode
                print(f"  Processing colors... {i}/{len(unique_colors)} ({i*100//len(unique_colors)}%)")
            
            # Skip pure black (background) if not including background
            if not self.include_background and np.array_equal(color, [0, 0, 0]):
                if show_progress:
                    print("    Skipping black background")
                continue
            
            # Create binary mask for this color
            mask = np.all(img_array == color, axis=2)
            
            # Skip if color barely present
            if np.sum(mask) < 10:
                continue
            
            # Convert to path using potrace
            result = self.trace_bitmap(mask, width, height)
            
            if result:
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                paths = result.get('paths', '')
                transform = result.get('transform', '')
                
                if paths:
                    if transform:
                        svg_parts.append(f'  <g transform="{transform}">')
                        svg_parts.append(f'    <path d="{paths}" fill="{hex_color}" />')
                        svg_parts.append('  </g>')
                    else:
                        svg_parts.append(f'  <path d="{paths}" fill="{hex_color}" />')
        
        # SVG footer
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def trace_bitmap(self, mask, target_width, target_height):
        """Use potrace to convert a binary mask to SVG path data.
        
        Args:
            mask: Boolean numpy array
            target_width: Desired output width in pixels
            target_height: Desired output height in pixels
        """
        tmp_bmp_path = None
        tmp_svg_path = None
        try:
            # Convert boolean mask to bitmap
            # Invert the mask since potrace treats black as foreground
            bitmap = Image.fromarray((~mask * 255).astype(np.uint8))
            
            # Save as temporary BMP
            with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as tmp_bmp:
                bitmap.save(tmp_bmp.name)
                tmp_bmp_path = tmp_bmp.name
            
            # Run potrace
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
                tmp_svg_path = tmp_svg.name
            
            # Run potrace without specifying dimensions - we'll handle scaling ourselves
            cmd = [
                'potrace',
                '-s',  # SVG output
                '-o', tmp_svg_path,
                tmp_bmp_path
            ]
            
            if self.simplify:
                cmd.extend(['-t', str(self.threshold)])  # Threshold
                cmd.extend(['-O', '0.2'])  # Optimization tolerance
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Potrace error: {result.stderr}")
                return None
            
            # Extract path data and dimensions from SVG
            tree = ET.parse(tmp_svg_path)
            root = tree.getroot()
            
            # Get the dimensions from potrace output
            svg_width = root.get('width', '0')
            svg_height = root.get('height', '0')
            viewbox = root.get('viewBox', '')
            
            # Parse dimensions (remove 'pt' if present)
            if svg_width.endswith('pt'):
                svg_width = float(svg_width[:-2])
            else:
                svg_width = float(svg_width) if svg_width else mask.shape[1]
            
            if svg_height.endswith('pt'):
                svg_height = float(svg_height[:-2])
            else:
                svg_height = float(svg_height) if svg_height else mask.shape[0]
            
            # Calculate scaling factors to match target dimensions
            scale_x = target_width / svg_width if svg_width > 0 else 1
            scale_y = target_height / svg_height if svg_height > 0 else 1
            
            # Collect all path data with proper scaling
            all_paths = []
            
            for elem in root.iter():
                if elem.tag.endswith('g') and 'transform' in elem.attrib:
                    # Found a group with transform
                    group_transform = elem.get('transform')
                    for path in elem.iter():
                        if path.tag.endswith('path'):
                            d = path.get('d')
                            if d:
                                # We'll apply scaling to the entire group
                                all_paths.append({
                                    'path': d,
                                    'transform': f'scale({scale_x},{scale_y}) {group_transform}'
                                })
                elif elem.tag.endswith('path'):
                    # Direct path without group
                    d = elem.get('d')
                    if d:
                        all_paths.append({
                            'path': d,
                            'transform': f'scale({scale_x},{scale_y})'
                        })
            
            # Combine all paths into a single path element with proper transform
            if all_paths:
                # If there's only one path with one transform, simplify
                if len(all_paths) == 1:
                    return {
                        'paths': all_paths[0]['path'],
                        'transform': all_paths[0]['transform']
                    }
                else:
                    # Multiple paths - combine them
                    combined_paths = ' '.join([p['path'] for p in all_paths])
                    # Use the first transform (they should all be the same)
                    return {
                        'paths': combined_paths,
                        'transform': all_paths[0]['transform'] if all_paths else ''
                    }
            
            return None
            
        except Exception as e:
            print(f"Error tracing bitmap: {e}")
            return None
        finally:
            # Always clean up temp files
            if tmp_bmp_path and os.path.exists(tmp_bmp_path):
                try:
                    os.unlink(tmp_bmp_path)
                except:
                    pass
            if tmp_svg_path and os.path.exists(tmp_svg_path):
                try:
                    os.unlink(tmp_svg_path)
                except:
                    pass


class SVGColorProcessor:
    """Process existing SVG files to reduce colors and remove gradients."""
    
    @staticmethod
    def rasterize_svg(svg_path, output_path=None, dpi=150):
        """Rasterize an SVG file to a PNG image.
        
        Args:
            svg_path: Path to input SVG file
            output_path: Optional path for output PNG (if None, returns PIL Image)
            dpi: Resolution for rasterization (higher = better quality but larger)
        
        Returns:
            PIL Image object if output_path is None, otherwise the output path
        """
        try:
            # Try using cairosvg if available (best quality)
            try:
                import cairosvg
                
                # Get SVG dimensions
                tree = ET.parse(svg_path)
                root = tree.getroot()
                
                # Try to extract width/height
                width = root.get('width', '800')
                height = root.get('height', '600')
                viewbox = root.get('viewBox', '')
                
                # Parse dimensions
                if viewbox:
                    parts = viewbox.split()
                    if len(parts) == 4:
                        width = float(parts[2])
                        height = float(parts[3])
                else:
                    # Remove units if present
                    width = float(re.sub(r'[^\d.]', '', str(width)) or 800)
                    height = float(re.sub(r'[^\d.]', '', str(height)) or 600)
                
                # Calculate scale based on DPI (assuming 96 DPI base)
                scale = dpi / 96.0
                output_width = int(width * scale)
                output_height = int(height * scale)
                
                if output_path:
                    cairosvg.svg2png(
                        url=svg_path,
                        write_to=output_path,
                        output_width=output_width,
                        output_height=output_height
                    )
                    return output_path
                else:
                    # Convert to bytes and return as PIL Image
                    png_bytes = cairosvg.svg2png(
                        url=svg_path,
                        output_width=output_width,
                        output_height=output_height
                    )
                    from io import BytesIO
                    return Image.open(BytesIO(png_bytes))
                    
            except ImportError:
                # Fallback: use Inkscape if available
                import subprocess
                import tempfile
                
                if output_path:
                    png_path = output_path
                else:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        png_path = tmp.name
                
                # Try Inkscape command
                try:
                    # Modern Inkscape (1.0+)
                    cmd = [
                        'inkscape',
                        svg_path,
                        '--export-type=png',
                        f'--export-filename={png_path}',
                        f'--export-dpi={dpi}'
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        # Try older Inkscape syntax
                        cmd = [
                            'inkscape',
                            svg_path,
                            '--export-png', png_path,
                            '--export-dpi', str(dpi)
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            raise Exception(f"Inkscape failed: {result.stderr}")
                    
                    if output_path:
                        return output_path
                    else:
                        # Load as PIL Image and clean up temp file
                        img = Image.open(png_path)
                        os.unlink(png_path)
                        return img
                        
                except FileNotFoundError:
                    # Final fallback: Simple conversion (less accurate but works)
                    print("Warning: Neither cairosvg nor Inkscape available. Using simple fallback rasterization.")
                    print("For better quality, install cairosvg (pip install cairosvg) or Inkscape.")
                    
                    # Use a simple approach: just create a basic raster from SVG size
                    tree = ET.parse(svg_path)
                    root = tree.getroot()
                    
                    # Extract dimensions
                    width = root.get('width', '800')
                    height = root.get('height', '600')
                    viewbox = root.get('viewBox', '')
                    
                    if viewbox:
                        parts = viewbox.split()
                        if len(parts) == 4:
                            width = float(parts[2])
                            height = float(parts[3])
                    else:
                        width = float(re.sub(r'[^\d.]', '', str(width)) or 800)
                        height = float(re.sub(r'[^\d.]', '', str(height)) or 600)
                    
                    # Create a blank image with message
                    scale = dpi / 96.0
                    img_width = int(width * scale)
                    img_height = int(height * scale)
                    
                    raise Exception("SVG rasterization requires cairosvg or Inkscape. "
                                    "Install with: pip install cairosvg (requires cairo library) "
                                    "or install Inkscape from https://inkscape.org")
                    
        except Exception as e:
            print(f"Error rasterizing SVG: {e}")
            raise
    
    @staticmethod
    def extract_gradient_colors(root):
        """Extract all colors from gradients in an SVG."""
        gradient_colors = []
        
        for elem in root.iter():
            if 'linearGradient' in elem.tag or 'radialGradient' in elem.tag:
                # Extract all stop colors from the gradient
                for stop in elem:
                    if 'stop' in stop.tag:
                        stop_color = None
                        stop_opacity = 1.0
                        
                        # Check stop-color attribute
                        stop_color = stop.get('stop-color')
                        
                        # Check style attribute for stop-color
                        style = stop.get('style', '')
                        if style and not stop_color:
                            match = re.search(r'stop-color:\s*([^;]+)', style)
                            if match:
                                stop_color = match.group(1)
                        
                        # Check for opacity
                        stop_opacity_attr = stop.get('stop-opacity')
                        if stop_opacity_attr:
                            try:
                                stop_opacity = float(stop_opacity_attr)
                            except:
                                pass
                        elif style:
                            match = re.search(r'stop-opacity:\s*([^;]+)', style)
                            if match:
                                try:
                                    stop_opacity = float(match.group(1))
                                except:
                                    pass
                        
                        # Add color if found (skip if too transparent)
                        if stop_color and stop_opacity > 0.1:
                            gradient_colors.append(stop_color)
        
        return gradient_colors
    
    @staticmethod
    def remove_gradients(svg_path, output_path):
        """Convert gradients to solid colors in an SVG file."""
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Find all gradients
        gradients = {}
        for elem in root.iter():
            if 'linearGradient' in elem.tag or 'radialGradient' in elem.tag:
                grad_id = elem.get('id')
                if grad_id:
                    # Get first stop color
                    for stop in elem:
                        if 'stop' in stop.tag:
                            stop_color = stop.get('stop-color') or stop.get('style', '')
                            if 'stop-color:' in stop_color:
                                stop_color = re.search(r'stop-color:\s*([^;]+)', stop_color).group(1)
                            if stop_color:
                                gradients[grad_id] = stop_color
                                break
        
        # Replace gradient references
        for elem in root.iter():
            for attr in ['fill', 'stroke']:
                value = elem.get(attr, '')
                for grad_id, color in gradients.items():
                    if f'url(#{grad_id})' in value:
                        elem.set(attr, color)
            
            # Process style attribute
            style = elem.get('style', '')
            if style:
                for grad_id, color in gradients.items():
                    style = re.sub(f'url\\(#{grad_id}\\)', color, style)
                elem.set('style', style)
        
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return output_path
    
    @staticmethod
    def quantize_colors(svg_path, output_path, n_colors=8, sample_gradients=False):
        """Reduce the number of colors in an SVG file.
        
        Args:
            svg_path: Input SVG file path
            output_path: Output SVG file path
            n_colors: Number of colors to quantize to
            sample_gradients: If True, sample colors from gradients instead of removing them
        """
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Extract all colors
        colors = set()
        color_pattern = re.compile(r'#[0-9a-fA-F]{3,6}')
        
        # Extract gradient colors if requested
        if sample_gradients:
            gradient_colors = SVGColorProcessor.extract_gradient_colors(root)
            for gc in gradient_colors:
                if gc and gc.startswith('#'):
                    colors.add(gc)
                elif gc and gc.startswith('rgb'):
                    # Convert rgb() format to hex
                    match = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', gc)
                    if match:
                        r, g, b = map(int, match.groups())
                        hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
                        colors.add(hex_color)
        
        for elem in root.iter():
            for attr in ['fill', 'stroke', 'stop-color']:
                value = elem.get(attr, '')
                if value and value.startswith('#'):
                    colors.add(value)
            
            style = elem.get('style', '')
            if style:
                for match in color_pattern.findall(style):
                    colors.add(match)
        
        if not colors:
            # No colors to process
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            return output_path
        
        # Convert to RGB for clustering
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        rgb_colors = [hex_to_rgb(c) for c in colors]
        
        # Cluster colors
        if len(rgb_colors) <= n_colors:
            palette = rgb_colors
        else:
            X = np.array(rgb_colors)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(X)
            palette = [tuple(int(x) for x in center) for center in kmeans.cluster_centers_]
        
        # Create color mapping
        color_map = {}
        for hex_color in colors:
            rgb = hex_to_rgb(hex_color)
            # Find closest palette color
            min_dist = float('inf')
            closest = palette[0]
            for p_color in palette:
                dist = sum((a-b)**2 for a, b in zip(rgb, p_color))
                if dist < min_dist:
                    min_dist = dist
                    closest = p_color
            color_map[hex_color] = '#{:02x}{:02x}{:02x}'.format(*closest)
        
        # Apply color mapping
        for elem in root.iter():
            for attr in ['fill', 'stroke', 'stop-color']:
                value = elem.get(attr, '')
                if value in color_map:
                    elem.set(attr, color_map[value])
            
            style = elem.get('style', '')
            if style:
                for old_color, new_color in color_map.items():
                    style = style.replace(old_color, new_color)
                elem.set('style', style)
        
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert images to printable SVGs with limited color palettes'
    )
    
    parser.add_argument('input', help='Input image or SVG file')
    parser.add_argument('output', nargs='?', help='Output SVG file (default: input_Ncolors.svg)')
    parser.add_argument(
        '-c', '--colors', type=int, default=8,
        help='Number of colors in palette (default: 8, 0 for unlimited)'
    )
    parser.add_argument(
        '-m', '--method', 
        choices=['kmeans', 'posterize', 'adaptive'],
        default='kmeans',
        help='Color quantization method (default: kmeans)'
    )
    parser.add_argument(
        '--no-simplify', action='store_true',
        help='Disable path simplification'
    )
    parser.add_argument(
        '-t', '--threshold', type=int, default=128,
        help='Threshold for bitmap tracing (0-255, default: 128)'
    )
    parser.add_argument(
        '--remove-gradients', action='store_true',
        help='Remove gradients from SVG (for SVG input)'
    )
    parser.add_argument(
        '--sample-gradients', action='store_true', default=None,
        help='Sample colors from gradients for quantization (default for SVG input)'
    )
    parser.add_argument(
        '--no-sample-gradients', action='store_true',
        help='Disable gradient color sampling (keep gradients as-is)'
    )
    parser.add_argument(
        '--rasterize', action='store_true',
        help='Rasterize SVG before processing (converts gradients to pixels, for SVG input)'
    )
    parser.add_argument(
        '--dpi', type=int, default=150,
        help='DPI for SVG rasterization (default: 150, higher = better quality)'
    )
    parser.add_argument(
        '--quantize-only', action='store_true',
        help='Only quantize colors in existing SVG (for SVG input)'
    )
    parser.add_argument(
        '--include-background', action='store_true',
        help='Include black background in output (default: skip black background)'
    )
    parser.add_argument(
        '--denoise', action='store_true',
        help='Apply denoising to reduce artifacts (good for AI-generated images)'
    )
    parser.add_argument(
        '--denoise-strength', type=int, default=3, choices=[1, 3, 5, 7],
        help='Denoising strength (1=light, 3=medium, 5=strong, 7=very strong, default: 3)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Generate default output filename if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        # Create output filename like "input_8colors.svg" or "input_unlimited.svg"
        if args.colors == 0:
            output_path = input_path.parent / f"{input_path.stem}_unlimited.svg"
        else:
            output_path = input_path.parent / f"{input_path.stem}_{args.colors}colors.svg"
        print(f"Output file: {output_path}")
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    # Check if input is SVG
    if input_path.suffix.lower() == '.svg':
        # Check for mutually exclusive options
        exclusive_count = sum([args.remove_gradients, args.rasterize, args.no_sample_gradients])
        if exclusive_count > 1:
            print("Error: Can only use one of --remove-gradients, --rasterize, or --no-sample-gradients")
            sys.exit(1)
        
        if args.rasterize:
            # Rasterize SVG first, then process as image
            print(f"Rasterizing SVG at {args.dpi} DPI...")
            processor = SVGColorProcessor()
            
            # Rasterize to PIL Image
            raster_image = processor.rasterize_svg(str(input_path), dpi=args.dpi)
            
            # Process as image
            print(f"Converting rasterized image to SVG with {args.colors} colors...")
            converter = ImageToSVGConverter(
                n_colors=args.colors,
                method=args.method,
                simplify=not args.no_simplify,
                threshold=args.threshold,
                include_background=args.include_background,
                denoise=args.denoise,
                denoise_strength=args.denoise_strength
            )
            
            # Save rasterized image to temp file for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                raster_image.save(tmp.name)
                temp_png = tmp.name
            
            try:
                converter.convert(temp_png, output_path)
            finally:
                os.unlink(temp_png)
        
        else:
            # Process existing SVG without rasterization
            processor = SVGColorProcessor()
            
            temp_path = input_path
            
            if args.remove_gradients:
                print("Removing gradients...")
                temp_output = output_path.with_suffix('.temp.svg')
                temp_path = processor.remove_gradients(temp_path, temp_output)
            
            # Determine if we should sample gradients
            # Default to sampling gradients unless explicitly disabled
            sample_gradients = not (args.remove_gradients or args.no_sample_gradients)
            
            if sample_gradients:
                print(f"Quantizing to {args.colors} colors (sampling gradient colors)...")
            else:
                print(f"Quantizing to {args.colors} colors...")
            
            processor.quantize_colors(temp_path, output_path, args.colors, sample_gradients=sample_gradients)
            
            if args.remove_gradients and temp_path != input_path:
                os.unlink(temp_path)
    
    else:
        # Convert image to SVG
        converter = ImageToSVGConverter(
            n_colors=args.colors,
            method=args.method,
            simplify=not args.no_simplify,
            threshold=args.threshold,
            include_background=args.include_background,
            denoise=args.denoise,
            denoise_strength=args.denoise_strength
        )
        converter.convert(input_path, output_path)
    
    print(f"Successfully created {output_path}")


if __name__ == '__main__':
    main()