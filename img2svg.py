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
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
# potrace is called via subprocess instead of Python bindings
import tempfile
import subprocess
import xml.etree.ElementTree as ET


class ImageToSVGConverter:
    """Main converter class for image to printable SVG conversion."""
    
    def __init__(self, n_colors=8, method='kmeans', simplify=True, threshold=128, include_background=False, suggested_colors=None):
        self.n_colors = n_colors
        self.method = method
        self.simplify = simplify
        self.threshold = threshold
        self.include_background = include_background
        self.suggested_colors = suggested_colors  # List of (R, G, B) tuples
        self.palette = None
    
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
        
        # Reduce colors
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
    
    def quantize_image(self, image):
        """Reduce the number of colors in an image."""
        # Convert to numpy array
        img_array = np.array(image)
        original_shape = img_array.shape
        
        # Reshape for clustering
        pixels = img_array.reshape(-1, 3)
        
        if self.method == 'kmeans':
            # Use k-means clustering
            if self.suggested_colors and len(self.suggested_colors) > 0:
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
        
        for i, color in enumerate(unique_colors):
            print(f"  Processing color {i+1}/{len(unique_colors)}: RGB{tuple(color)}")
            
            # Skip pure black (background) if not including background
            if not self.include_background and np.array_equal(color, [0, 0, 0]):
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
    def quantize_colors(svg_path, output_path, n_colors=8):
        """Reduce the number of colors in an SVG file."""
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Extract all colors
        colors = set()
        color_pattern = re.compile(r'#[0-9a-fA-F]{3,6}')
        
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
    parser.add_argument('output', help='Output SVG file')
    parser.add_argument(
        '-c', '--colors', type=int, default=8,
        help='Number of colors in palette (default: 8)'
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
        '--quantize-only', action='store_true',
        help='Only quantize colors in existing SVG (for SVG input)'
    )
    parser.add_argument(
        '--include-background', action='store_true',
        help='Include black background in output (default: skip black background)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    # Check if input is SVG
    if input_path.suffix.lower() == '.svg':
        # Process existing SVG
        processor = SVGColorProcessor()
        
        temp_path = input_path
        
        if args.remove_gradients:
            print("Removing gradients...")
            temp_output = output_path.with_suffix('.temp.svg')
            temp_path = processor.remove_gradients(temp_path, temp_output)
        
        print(f"Quantizing to {args.colors} colors...")
        processor.quantize_colors(temp_path, output_path, args.colors)
        
        if args.remove_gradients and temp_path != input_path:
            os.unlink(temp_path)
    
    else:
        # Convert image to SVG
        converter = ImageToSVGConverter(
            n_colors=args.colors,
            method=args.method,
            simplify=not args.no_simplify,
            threshold=args.threshold,
            include_background=args.include_background
        )
        converter.convert(input_path, output_path)
    
    print(f"Successfully created {output_path}")


if __name__ == '__main__':
    main()