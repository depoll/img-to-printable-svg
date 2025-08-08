import xml.etree.ElementTree as ET
import re

def convert_gradients_to_solid(svg_file, output_file):
    # Parse SVG
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Define namespaces
    namespaces = {
        'svg': 'http://www.w3.org/2000/svg',
        'xlink': 'http://www.w3.org/1999/xlink'
    }
    
    # Find all gradients and get their first stop color
    gradients = {}
    for gradient in root.findall('.//svg:linearGradient', namespaces) + \
                   root.findall('.//svg:radialGradient', namespaces):
        gradient_id = gradient.get('id')
        if gradient_id:
            # Get first stop color
            stops = gradient.findall('.//svg:stop', namespaces)
            if stops:
                style = stops[0].get('style', '')
                stop_color = stops[0].get('stop-color')
                stop_opacity = stops[0].get('stop-opacity', '1')
                
                # Extract color from style attribute if not in stop-color
                if not stop_color and style:
                    color_match = re.search(r'stop-color:\s*([^;]+)', style)
                    if color_match:
                        stop_color = color_match.group(1)
                
                if stop_color:
                    gradients[gradient_id] = (stop_color, stop_opacity)
    
    # Replace gradient references with solid colors
    for elem in root.iter():
        style = elem.get('style', '')
        fill = elem.get('fill', '')
        stroke = elem.get('stroke', '')
        
        # Check fill attribute
        for grad_id, (color, opacity) in gradients.items():
            if f'url(#{grad_id})' in fill:
                elem.set('fill', color)
                if float(opacity) < 1:
                    elem.set('fill-opacity', opacity)
        
        # Check stroke attribute
        for grad_id, (color, opacity) in gradients.items():
            if f'url(#{grad_id})' in stroke:
                elem.set('stroke', color)
                if float(opacity) < 1:
                    elem.set('stroke-opacity', opacity)
        
        # Check and update style attribute
        if style:
            for grad_id, (color, opacity) in gradients.items():
                style = re.sub(f'fill:\s*url\(#{grad_id}\)[^;]*', f'fill: {color}', style)
                style = re.sub(f'stroke:\s*url\(#{grad_id}\)[^;]*', f'stroke: {color}', style)
            elem.set('style', style)
    
    # Remove gradient definitions (optional)
    for gradient in root.findall('.//svg:linearGradient', namespaces) + \
                   root.findall('.//svg:radialGradient', namespaces):
        parent = root.find('.//*[@id="' + gradient.get('id') + '"]/..')
        if parent is not None:
            parent.remove(gradient)
    
    # Save modified SVG
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

# Usage
convert_gradients_to_solid('input.svg', 'output.svg')