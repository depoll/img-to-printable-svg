#!/bin/bash

echo "Fixing dependency compatibility issues..."
echo ""
echo "Option 1: Downgrade NumPy to compatible version"
echo "Run: pip install 'numpy<2.0'"
echo ""
echo "Option 2: Upgrade scikit-learn"
echo "Run: pip install --upgrade scikit-learn"
echo ""
echo "Option 3: Use conda to manage dependencies"
echo "Run: conda update scikit-learn"
echo ""
echo "Recommended: Try option 1 first (downgrade numpy)"