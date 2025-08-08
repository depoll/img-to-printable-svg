import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import colorsys

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except:
        return (0, 0, 0)

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def color_distance(c1, c2):
    """Calculate Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def quantize_color(color, palette):
    """Find the closest color in the palette."""
    min_distance = float('inf')
    closest_color = palette[0]
    
    for palette_color in palette:
        distance = color_distance(color, palette_color)
        if distance < min_distance:
            min_distance = distance
            closest_color = palette_color
    
    return closest_color

def extract_colors_from_svg(root):
    """Extract all unique colors from SVG."""
    colors = set()
    color_pattern = re.compile(r'#[0-9a-fA-F]{3,6}|rgb\s*\([^)]+\)|[a-z]+')
    
    for elem in root.iter():
        # Check various color attributes
        for attr in ['fill', 'stroke', 'stop-color', 'flood-color', 'lighting-color']:
            value = elem.get(attr, '')
            if value and value != 'none' and not value.startswith('url('):
                colors.add(value)
        
        # Check style attribute
        style = elem.get('style', '')
        if style:
            matches = color_pattern.findall(style)
            for match in matches:
                if match and not match.startswith('url(') and match != 'none':
                    colors.add(match)
    
    return colors

def parse_color(color_str):
    """Parse various color formats to RGB."""
    color_str = color_str.strip()
    
    # Hex colors
    if color_str.startswith('#'):
        return hex_to_rgb(color_str)
    
    # RGB colors
    if color_str.startswith('rgb'):
        matches = re.findall(r'\d+', color_str)
        if len(matches) >= 3:
            return tuple(int(m) for m in matches[:3])
    
    # Named colors (basic set)
    named_colors = {
        'black': (0, 0, 0), 'white': (255, 255, 255),
        'red': (255, 0, 0), 'green': (0, 128, 0),
        'blue': (0, 0, 255), 'yellow': (255, 255, 0),
        'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
        'gray': (128, 128, 128), 'grey': (128, 128, 128),
        'silver': (192, 192, 192), 'maroon': (128, 0, 0),
        'olive': (128, 128, 0), 'lime': (0, 255, 0),
        'aqua': (0, 255, 255), 'teal': (0, 128, 128),
        'navy': (0, 0, 128), 'fuchsia': (255, 0, 255),
        'purple': (128, 0, 128), 'orange': (255, 165, 0)
    }
    
    if color_str.lower() in named_colors:
        return named_colors[color_str.lower()]
    
    return (0, 0, 0)  # Default to black

def generate_palette(colors, n_colors, method='kmeans'):
    """Generate a color palette from existing colors."""
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Convert colors to RGB
    rgb_colors = [parse_color(c) for c in colors if c and c != 'none']
    
    if len(rgb_colors) <= n_colors:
        return rgb_colors
    
    # Use k-means clustering to find representative colors
    X = np.array(rgb_colors)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Get cluster centers as palette
    palette = [tuple(int(x) for x in center) for center in kmeans.cluster_centers_]
    return palette

def quantize_svg_colors(svg_file, output_file, n_colors=8, palette=None, method='kmeans'):
    """
    Quantize colors in an SVG file.
    
    Args:
        svg_file: Input SVG file path
        output_file: Output SVG file path
        n_colors: Number of colors in the quantized palette
        palette: Custom palette (list of RGB tuples). If None, generates from existing colors
        method: Method for generating palette ('kmeans', 'uniform', 'posterize')
    """
    # Parse SVG
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Extract all colors from SVG
    original_colors = extract_colors_from_svg(root)
    print(f"Found {len(original_colors)} unique colors in SVG")
    
    # Generate or use provided palette
    if palette is None:
        if method == 'uniform':
            # Create uniform palette across color space
            palette = []
            steps = int(n_colors ** (1/3)) + 1
            for r in range(0, 256, 256 // steps):
                for g in range(0, 256, 256 // steps):
                    for b in range(0, 256, 256 // steps):
                        palette.append((r, g, b))
            palette = palette[:n_colors]
        elif method == 'posterize':
            # Posterize effect - reduce bit depth
            levels = max(2, int(256 / (256 / n_colors) ** (1/3)))
            palette = []
            for r in range(0, 256, 256 // levels):
                for g in range(0, 256, 256 // levels):
                    for b in range(0, 256, 256 // levels):
                        palette.append((r, g, b))
            # Sample down if too many colors
            if len(palette) > n_colors:
                step = len(palette) // n_colors
                palette = palette[::step][:n_colors]
        else:  # kmeans
            palette = generate_palette(original_colors, n_colors, method)
    
    print(f"Using palette with {len(palette)} colors")
    
    # Create color mapping
    color_map = {}
    for color_str in original_colors:
        if color_str and color_str != 'none':
            rgb = parse_color(color_str)
            quantized_rgb = quantize_color(rgb, palette)
            color_map[color_str] = rgb_to_hex(quantized_rgb)
    
    # Apply color quantization to SVG
    for elem in root.iter():
        # Process color attributes
        for attr in ['fill', 'stroke', 'stop-color', 'flood-color', 'lighting-color']:
            value = elem.get(attr, '')
            if value in color_map:
                elem.set(attr, color_map[value])
        
        # Process style attribute
        style = elem.get('style', '')
        if style:
            for original, quantized in color_map.items():
                # Escape special regex characters
                escaped = re.escape(original)
                style = re.sub(f'fill:\\s*{escaped}', f'fill: {quantized}', style)
                style = re.sub(f'stroke:\\s*{escaped}', f'stroke: {quantized}', style)
                style = re.sub(f'stop-color:\\s*{escaped}', f'stop-color: {quantized}', style)
            elem.set('style', style)
    
    # Save modified SVG
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Print color mapping
    print("\nColor mapping:")
    for original, quantized in sorted(color_map.items()):
        print(f"  {original} -> {quantized}")

# Example usage with different options:

# 1. Basic usage - auto-generate palette with k-means
quantize_svg_colors('input.svg', 'output_8colors.svg', n_colors=8)

# 2. Use uniform palette
quantize_svg_colors('input.svg', 'output_uniform.svg', n_colors=16, method='uniform')

# 3. Use posterize effect
quantize_svg_colors('input.svg', 'output_posterize.svg', n_colors=8, method='posterize')

# 4. Use custom palette (e.g., retro gaming palette)
custom_palette = [
    (0, 0, 0),        # Black
    (255, 255, 255),  # White
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
]
quantize_svg_colors('input.svg', 'output_custom.svg', palette=custom_palette)

# 5. Create a monochrome version (shades of single color)
def create_monochrome_palette(base_color, n_shades):
    """Create a monochrome palette with n shades of a base color."""
    palette = []
    for i in range(n_shades):
        factor = i / (n_shades - 1)
        shade = tuple(int(c * factor) for c in base_color)
        palette.append(shade)
    return palette

blue_palette = create_monochrome_palette((0, 100, 255), 8)
quantize_svg_colors('input.svg', 'output_monochrome.svg', palette=blue_palette)