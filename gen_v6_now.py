from roboflow import Roboflow
import time

rf = Roboflow(api_key='CXCNe93GGPiEeF2XrcYE')
proj = rf.workspace('fire-cjxu1').project('find-glioma-and-brainscope-ai')

print('Generating v6 from existing dataset...')
ver_num = proj.generate_version(settings={
    'augmentation': {
        'bbFlipX': {'percent': 50},
        'crop': {'min': 0, 'max': 15},
        'rotation': {'degrees': 15},
        'brightness': {'percent': 15, 'brighten': True, 'darken': True},
        'blur': {'pixels': 2},
        'noise': {'percent': 1}
    },
    'preprocessing': {
        'auto-orient': True,
        'resize': {'width': 640, 'height': 640, 'format': 'Stretch to'}
    }
})
print('Version generated:', ver_num)

time.sleep(5)

# Try training with progressively smaller models
for model in ['rf-detr-l', 'rf-detr-large', 'rf-detr-medium', 'yolov11x', 'yolov11l', 'yolov8x']:
    try:
        v = proj.version(ver_num if isinstance(ver_num, int) else 6)
        v.train(model_type=model, speed='accurate')
        print(f'Training started with {model}!')
        break
    except Exception as e:
        print(f'{model} failed: {e}')
