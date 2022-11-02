config = {
        'BATCH_SIZE': 2,
        'constant_fields': ['sic', 'sst', 'sic_trend'],
        'dated_fields': ['t2m', 'xwind', 'ywind'],
        'train_augment': False,
        'val_augment': False,
        'learning_rate': 0.1,
        'epochs': 20,
        'pooling_factor': 4,
        'channels': [64, 128, 256, 512],
        'height': 1792,
        'width': 1792,
        'lower_boundary': 578,
        'rightmost_boundary': 1792
    }