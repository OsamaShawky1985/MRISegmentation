# Quick Answer: YOLOv10 Pretrained Weights

## 🎯 Your Question: "What pretrained weights does YOLOv10 use? COCO dataset or other weights?"

## ✅ **Answer: YES, our YOLOv10 implementation uses COCO dataset pretrained weights**

### Key Points:

1. **Default Behavior**: YOLOv10 loads COCO pretrained weights automatically
   ```python
   self.model = YOLO(f'{self.model_size}.pt')  # Downloads COCO weights
   ```

2. **COCO Dataset Info**:
   - **80 object classes** (person, car, bicycle, etc.)
   - **330K images** with 2.5M object instances
   - Industry standard for object detection

3. **Adaptation for Brain Tumors**:
   - Starts with 80 COCO classes
   - Adapts to 1 class (brain tumor) during training
   - Fine-tunes on your medical data

4. **Why COCO Works Well**:
   - ✅ General visual features (edges, shapes, boundaries)
   - ✅ Faster training convergence
   - ✅ Better performance than training from scratch
   - ✅ Requires less medical training data

5. **Available Models** (all COCO pretrained):
   - `yolov10n.pt` - Nano (2.3M params)
   - `yolov10s.pt` - Small (7.2M params)  
   - `yolov10m.pt` - Medium (15.4M params)
   - `yolov10l.pt` - Large (24.4M params)
   - `yolov10x.pt` - Extra Large (29.5M params)

### Configuration in Our Pipeline:
```yaml
yolo:
  model_size: "yolov10n"
  pretrained: true  # Uses COCO pretrained weights
  num_classes: 1    # Adapts from 80 COCO classes to 1 brain tumor class
```

### Where Weights Are Stored:
- **Download location**: `~/.cache/ultralytics/yolov10n.pt`
- **Custom trained**: `results/training/yolo_best.pt`

## 📖 Full Details:
See `docs/YOLOv10_Pretrained_Weights_Info.md` for comprehensive information about COCO dataset, transfer learning benefits, and customization options.
