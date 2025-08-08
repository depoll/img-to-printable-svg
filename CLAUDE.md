# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image to Printable SVG Converter - A utility for converting full-color raster images to SVGs with limited color palettes suitable for 3D printing, laser cutting, and other fabrication methods that require discrete colors.

## Core Architecture

### Main Components

1. **img2svg.py** - Unified command-line utility that handles:
   - Raster image to SVG conversion with color quantization
   - SVG color palette reduction
   - Gradient removal from existing SVGs
   - Multiple quantization algorithms (k-means, posterize, adaptive)

2. **ImageToSVGConverter** class - Core conversion logic:
   - Loads and quantizes raster images using configurable methods
   - Converts each color layer to vector paths using potrace
   - Combines layers into a single SVG output

3. **SVGColorProcessor** class - SVG manipulation utilities:
   - Removes gradients by converting to solid colors
   - Quantizes existing SVG colors to limited palette
   - Preserves SVG structure while modifying colors

### Original Scripts (reference implementations)

- `original_scripts/quantize_svg.py` - Standalone SVG color quantization
- `original_scripts/remove_gradients.py` - Gradient to solid color conversion

## Key Technical Details

### Color Quantization Methods

1. **k-means**: Uses scikit-learn's KMeans clustering to find optimal color palette
2. **posterize**: Simple bit-depth reduction for artistic effect
3. **adaptive**: Uses PIL's adaptive palette generation with median cut

### Image to Vector Conversion

- Uses potrace for bitmap tracing (converts binary masks to SVG paths)
- Processes each color as a separate layer
- Combines all layers into a single SVG with proper color fills

### Dependencies

- **Pillow**: Image loading and manipulation
- **numpy**: Array operations for color processing
- **scikit-learn**: K-means clustering for color quantization
- **potrace**: External command-line tool (must be installed separately via system package manager)

## Common Commands

### Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install potrace (platform-specific)
# macOS:
brew install potrace

# Ubuntu/Debian:
sudo apt-get install potrace

# Windows: Download from http://potrace.sourceforge.net/
```

### Basic Usage

```bash
# Convert image to SVG with 8 colors
python img2svg.py input.jpg output.svg

# Specify number of colors
python img2svg.py input.png output.svg -c 16

# Use different quantization method
python img2svg.py input.jpg output.svg -m posterize

# Process existing SVG - remove gradients and reduce colors
python img2svg.py input.svg output.svg --remove-gradients -c 4

# Only quantize colors in existing SVG
python img2svg.py input.svg output.svg --quantize-only -c 8
```

### Development and Testing

```bash
# Run with verbose output (add to img2svg.py if needed)
python img2svg.py input.jpg output.svg -v

# Test different color counts
for i in 2 4 8 16; do
    python img2svg.py test.jpg "test_${i}colors.svg" -c $i
done

# Process all images in a directory
for img in images/*.{jpg,png}; do
    python img2svg.py "$img" "output/$(basename ${img%.*}).svg"
done
```

## Project Structure

```
img-to-printable-svg/
├── img2svg.py              # Main utility script
├── requirements.txt        # Python dependencies
├── CLAUDE.md              # This file
├── README.md              # User documentation
└── original_scripts/      # Reference implementations
    ├── quantize_svg.py    # SVG color quantization
    └── remove_gradients.py # Gradient removal
```

## Important Implementation Notes

1. **Potrace Dependency**: The image-to-SVG conversion requires potrace to be installed system-wide. Without it, only SVG-to-SVG processing will work.

2. **Color Space**: All color processing is done in RGB space. For better perceptual results, consider implementing LAB color space quantization.

3. **Path Optimization**: The `simplify` option controls potrace's path simplification. Lower thresholds produce more detailed but larger files.

4. **Memory Usage**: Large images with many colors may use significant memory during k-means clustering. Consider adding batch processing for very large images.

5. **SVG Compatibility**: Output SVGs are standard-compliant and should work with all major vector graphics software and fabrication tools.

## Common Issues and Solutions

1. **"potrace: command not found"**: Install potrace using your system's package manager
2. **Large output files**: Increase threshold value or reduce number of colors
3. **Lost details**: Decrease threshold value or increase number of colors
4. **Memory errors**: Use posterize method instead of kmeans for large images

## Future Enhancements to Consider

- Add support for custom color palettes (specific Pantone/RAL colors)
- Implement dithering options for better gradient representation
- Add SVG optimization pass to reduce file size
- Support for multi-layer SVG output (one layer per color)
- Web interface for easier non-technical use
- Batch processing with progress indicators
- Color palette preview/export functionality