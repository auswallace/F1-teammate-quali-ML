# Circuit SVGs (Attribution)

Place circuit outline SVG files here (one per circuit), e.g., `AUS.svg`, `MON.svg`, `BHR.svg`.

## 📁 File Naming Convention
- Use circuit abbreviations: `AUS.svg`, `MON.svg`, `BHR.svg`
- Match the keys in `config/circuits_map.yaml`
- Keep filenames lowercase for consistency

## 🎯 Recommended Sources
- **Wikimedia Commons** (CC-BY-SA license)
- **Official F1 circuit diagrams** (if available)
- **Community-created outlines** (with proper attribution)

## 📋 Required SVG Structure
Each SVG should contain:
- `<path d="...">` elements with the circuit outline
- Clean, simplified paths (avoid complex fills or effects)
- Reasonable viewBox dimensions

## 🔗 Attribution
When downloading from Wikimedia Commons or other sources, add an attribution comment at the top of each SVG file:

```xml
<!-- 
Circuit outline from Wikimedia Commons
Source: [URL]
License: CC-BY-SA 4.0
Attribution: [Author Name]
-->
```

## 🚀 Usage
The React dashboard will automatically:
1. Load the SVG from this directory
2. Extract the path data
3. Render with neon stroke effects
4. Scale to fit the viewport

## 📝 Example Files
- `AUS.svg` - Albert Park Circuit (Melbourne)
- `MON.svg` - Circuit de Monaco
- `BHR.svg` - Bahrain International Circuit
- `GBR_SILVERSTONE.svg` - Silverstone Circuit
