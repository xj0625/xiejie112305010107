from ultralytics import YOLO
import os

model_path = 'exp-2.pt'
print(f'File exists: {os.path.exists(model_path)}')
print(f'File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB')

try:
    model = YOLO(model_path)
    print(f'Model loaded successfully')
    if hasattr(model.model, 'nc'):
        print(f'Number of classes: {model.model.nc}')
    else:
        print('Could not determine number of classes')
except Exception as e:
    print(f'Error loading model: {e}')