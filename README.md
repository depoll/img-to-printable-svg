# Image to Printable SVG Converter

Convert full-color images to SVGs with limited color palettes, perfect for 3D printing, laser cutting, vinyl cutting, screen printing, and other fabrication methods that require discrete colors.

## Features

- 🎨 **Color Quantization**: Reduce images to a specific number of colors (2-256)
- 🔄 **Multiple Methods**: Choose from k-means clustering, posterization, or adaptive quantization
- 📐 **Vector Conversion**: Automatically trace bitmap layers to create clean SVG paths
- 🎯 **SVG Processing**: Reduce colors in existing SVGs and remove gradients
- ⚡ **Optimized Output**: Control path simplification for file size vs. detail balance

## Installation

### Method 1: Docker (Recommended)

The easiest way to use this tool is with Docker. No need to install Python or Potrace manually!

**Using Docker Compose:**
```bash
# Pull the latest image
docker-compose pull

# Process an image (place files in ./input directory)
docker-compose run --rm img2svg /app/input/photo.jpg /app/output/result.svg -c 8

# Or build locally
docker-compose build
```

**Using Docker directly:**
```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/depoll/img-to-printable-svg:latest

# Run the converter
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  ghcr.io/depoll/img-to-printable-svg:latest \
  /app/input/photo.jpg /app/output/result.svg -c 8
```

### Method 2: Local Installation

#### Prerequisites

1. Python 3.8 or higher
2. Potrace (for image to vector conversion)
3. Cairo (optional, for SVG rasterization features)

#### Install Potrace

**macOS:**
```bash
brew install potrace
```

**Ubuntu/Debian:**
```bash
sudo apt-get install potrace
```

**Windows:**
Download from [Potrace website](http://potrace.sourceforge.net/)

#### Install Cairo (Optional - for SVG rasterization)

Cairo is needed if you want to use the `--rasterize` option for processing complex SVG gradients.

**macOS:**
```bash
brew install cairo
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libcairo2-dev
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install cairo-devel
# or for older versions:
sudo yum install cairo-devel
```

**Windows (using Conda):**
```bash
conda install cairo
```

**Arch Linux:**
```bash
sudo pacman -S cairo
```

**Alpine Linux:**
```bash
apk add cairo-dev
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install Pillow numpy scikit-learn
```

## Quick Start

### Using Docker

Create input and output directories:
```bash
mkdir -p input output
```

Place your images in the `input` directory, then run:
```bash
# Using docker-compose
docker-compose run --rm img2svg /app/input/photo.jpg /app/output/result.svg -c 8

# Or using docker directly
docker run --rm -v $(pwd):/app/work ghcr.io/depoll/img-to-printable-svg:latest \
  /app/work/input/photo.jpg /app/work/output/result.svg -c 8
```

### Using Local Installation

Convert a photo to an 8-color SVG:
```bash
python img2svg.py photo.jpg output.svg
```

### Specify Number of Colors

Create a 4-color version for screen printing:
```bash
python img2svg.py design.png screenprint.svg -c 4
```

### Different Quantization Methods

**K-means (default - best for photos):**
```bash
python img2svg.py photo.jpg output.svg -m kmeans
```

**Posterize (artistic effect):**
```bash
python img2svg.py art.jpg posterized.svg -m posterize
```

**Adaptive (good for illustrations):**
```bash
python img2svg.py illustration.png output.svg -m adaptive
```

### Process Existing SVGs

**Remove gradients and reduce colors:**
```bash
python img2svg.py complex.svg simple.svg --remove-gradients -c 8
```

**Sample gradient colors for better quantization:**
```bash
python img2svg.py gradient.svg output.svg --sample-gradients -c 8
```

**Rasterize SVG first (best for complex gradients):**
```bash
python img2svg.py complex.svg output.svg --rasterize -c 8 --dpi 150
```

**Only reduce colors (keep gradients):**
```bash
python img2svg.py input.svg output.svg --quantize-only -c 16
```

## Command Line Options

```
usage: img2svg.py [-h] [-c COLORS] [-m {kmeans,posterize,adaptive}]
                  [--no-simplify] [-t THRESHOLD] [--remove-gradients]
                  [--sample-gradients] [--rasterize] [--dpi DPI]
                  [--quantize-only] [--include-background] input output

arguments:
  input                 Input image or SVG file
  output               Output SVG file

options:
  -h, --help           Show help message
  -c, --colors         Number of colors in palette (default: 8)
  -m, --method         Color quantization method (default: kmeans)
  --no-simplify        Disable path simplification (more detail, larger files)
  -t, --threshold      Threshold for bitmap tracing (0-255, default: 128)
  --remove-gradients   Remove gradients from SVG input
  --sample-gradients   Sample colors from gradients for quantization
  --rasterize          Rasterize SVG before processing (requires Cairo)
  --dpi                DPI for SVG rasterization (default: 150)
  --quantize-only      Only quantize colors in existing SVG
  --include-background Include black background in output
```

## Use Cases

### 3D Printing (Multi-Material)
Create SVGs with distinct colors for each material:
```bash
python img2svg.py logo.png logo_3d.svg -c 3
```

### Screen Printing
Generate separations with limited colors:
```bash
python img2svg.py design.jpg print_ready.svg -c 4 -m posterize
```

### Vinyl Cutting
Create simple, cuttable designs:
```bash
python img2svg.py graphic.png vinyl.svg -c 2 -t 200
```

### Laser Engraving
Convert photos to engraveable vectors:
```bash
python img2svg.py portrait.jpg engrave.svg -c 4 --no-simplify
```

## Tips for Best Results

1. **Start with high-quality images**: Better input = better output
2. **Experiment with color counts**: Try different values to find the sweet spot
3. **Adjust threshold**: Higher values (200+) for simpler shapes, lower (50-100) for detail
4. **Use appropriate methods**:
   - `kmeans`: Best for photographs and natural images
   - `posterize`: Good for artistic/graphic effects
   - `adaptive`: Works well for illustrations and logos

## Troubleshooting

**"potrace: command not found"**
- Install potrace using your system's package manager (see Installation)

**Output file is too large**
- Increase threshold value: `-t 200`
- Reduce number of colors: `-c 4`
- Enable simplification (remove `--no-simplify` if used)

**Lost too much detail**
- Decrease threshold value: `-t 50`
- Increase number of colors: `-c 16`
- Disable simplification: `--no-simplify`

**Memory errors with large images**
- Use posterize method instead of kmeans: `-m posterize`
- Resize image before processing
- Process in smaller sections

## Examples

### Create a 2-color stencil
```bash
python img2svg.py face.jpg stencil.svg -c 2 -t 150
```

### Convert logo to 5 brand colors
```bash
python img2svg.py company_logo.png brand_logo.svg -c 5 -m kmeans
```

### Prepare image for t-shirt printing
```bash
python img2svg.py design.jpg tshirt.svg -c 6 -m posterize
```

## License

MIT License - Feel free to use this tool for personal and commercial projects.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- Built with [Potrace](http://potrace.sourceforge.net/) for bitmap tracing
- Uses [scikit-learn](https://scikit-learn.org/) for k-means clustering
- Powered by [Pillow](https://python-pillow.org/) for image processing