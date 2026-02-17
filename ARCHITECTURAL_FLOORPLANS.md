# Working with Architectural Floor Plans - Important Notes

## Current Limitation

The converter currently works best with **simple floor plans** where:
- Rooms are **separate white regions** surrounded by black walls
- Walls are **thick and continuous**
- There is **minimal text or annotations**

## Why Your Floor Plan Shows Only One Room

Your architectural floor plan (`example4.png`) has:

1. **Black walls on white background** - The entire floor is one continuous white region
2. **Interior walls** that divide the space but don't create separate enclosed white regions
3. **Text labels and dimensions** that interfere with detection
4. **Thin walls** (2-3 pixels) vs. the expected 10+ pixels

### What the Algorithm Sees

```
Current detection:
┌─────────────────────────┐
│  ONE LARGE WHITE SPACE  │  ← Detected as single room
│  (entire floor plan)    │
└─────────────────────────┘

What we want:
┌──────┬─────┬──────┐
│ Bed  │Bath │ Bed  │  ← Separate rooms
├──────┼─────┴──────┤
│Kitchen│ Living Rm  │
└──────┴────────────┘
```

## Solutions

### Option 1: Manual Image Preparation (Recommended for Now)

Manually edit your floor plan in an image editor to:

1. **Fill each room with a different shade** or keep rooms white but make walls VERY thick (15-20 pixels)
2. **Remove all text labels** and dimensions
3. **Ensure walls are continuous** with no gaps

Example using Paint/GIMP/Photoshop:
- Make walls 15-20 pixels thick
- Fill walls with solid black
- Remove all text
- Save as PNG

### Option 2: Use the Preprocessing Script

I've created `preprocess_architectural.py` which helps but has limitations:

```bash
# Step 1: Preprocess to remove text
python preprocess_architectural.py input/example4.png preprocessed/

# Step 2: Run converter with architectural config
python main.py --input preprocessed --output output --config config_architectural.yaml
```

**Note**: This removes text but doesn't solve the "one white region" problem.

### Option 3: Advanced Algorithm (Future Enhancement)

To properly handle architectural floor plans, we would need:

1. **Flood Fill Algorithm**: Start from room centers and fill outward until hitting walls
2. **Watershed Segmentation**: Treat walls as barriers and segment the floor plan
3. **Deep Learning**: Train a model to recognize room boundaries
4. **Manual Room Marking**: Allow users to click room centers

## Immediate Workaround

For your specific floor plan, here's what you can do:

### Quick Manual Fix

1. Open `input/example4.png` in an image editor
2. Use the **paint bucket** tool to fill each room with white
3. Use the **pencil/brush** tool (15px width) to:
   - Trace over all walls in black
   - Make sure walls are thick and continuous
4. **Erase all text** labels and dimensions
5. Save as `example4_manual.png`
6. Run: `python main.py --input input --output output`

### Example of What Works Well

The simple floor plan I generated (`sample_floorplan.png`) works because:
- ✅ Thick walls (10 pixels)
- ✅ No text or annotations
- ✅ Clean black and white
- ✅ Continuous wall lines

## Configuration Tips

If you manually prepare your floor plan, use these settings in `config.yaml`:

```yaml
image_processing:
  threshold: 127
  adaptive_threshold: false  # Use global threshold for clean images
  blur_kernel_size: 3
  morph_kernel_size: 2

room_detection:
  min_area: 2000  # Adjust based on your room sizes
  contour_epsilon: 0.01  # Lower = more detailed polygons
```

## Future Enhancements Needed

To properly support architectural floor plans like yours, we would need to implement:

1. **Watershed Segmentation**
   ```python
   # Detect wall centers
   # Apply distance transform
   # Use watershed to segment rooms
   ```

2. **Connected Components with Wall Detection**
   ```python
   # Invert image (walls become white)
   # Find wall skeleton
   # Use walls as boundaries for room detection
   ```

3. **Interactive Mode**
   ```python
   # Let user click on each room center
   # Flood fill from those points
   # Stop at wall boundaries
   ```

## Summary

**Current Status**:
- ✅ Works perfectly with simple, clean floor plans
- ⚠️ Limited support for complex architectural drawings
- ❌ Cannot automatically separate rooms in plans where the entire floor is one white region

**Recommended Approach**:
1. For now, manually prepare floor plans (thick walls, no text)
2. Or use simple floor plan drawings
3. Future: Implement watershed/flood-fill algorithms for automatic room separation

## Example Workflow

```bash
# For simple floor plans (like sample_floorplan.png)
python main.py --input input --output output

# For architectural plans (requires manual preparation first)
# 1. Edit in image editor: thick walls, no text
# 2. Save to input/
python main.py --input input --output output --config config_architectural.yaml
```

## Need Help?

If you'd like me to implement the watershed segmentation or interactive room selection features, let me know! These would significantly improve support for real architectural floor plans.
