# ------------------ #
# PARAMETERS CHOICES #
# ------------------ #

parameters_choices = {
    'mode': ['train', 'resume', 'test'],
    'GPU': ['V100', 'A100', 'GTX'],
    'norm': ['none', 'min-max', 'std'],
    'backbone': ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152'],
    'optimizer': ['Adam', 'SGD'],
    'scheduler': ['ReduceLROnPlateau', 'StepLR', 'CosineAnnealing'],
    'loss': ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'SigmoidFocalLoss', 'FocalLoss'],
}