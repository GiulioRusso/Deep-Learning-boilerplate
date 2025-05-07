# ------------------ #
# PARAMETERS DEFAULT #
# ------------------ #

parameters_default = {

    # ------- #
    # DATASET #
    # ------- #
    'dataset': 'MyDataset',
    'num_classes': 10,
    'image_height': 512,
    'image_width': 512,
    'split': 'example',
    'norm': 'none',
    'do_dataset_augmentation': False,

    # ------ #
    # DEVICE #
    # ------ #
    'GPU': 'None',
    'num_threads': 32,

    # --------------- #
    # REPRODUCIBILITY #
    # --------------- #
    'seed': 0,

    # ----------- #
    # DATA LOADER #
    # ----------- #
    'batch_size_train': 32,
    'batch_size_val': 32,
    'batch_size_test': 32,

    'num_workers': 8,

    # ------- #
    # NETWORK #
    # ------- #
    'network_name': 'MyNetwork',
    'backbone': 'ResNet-18',
    'pretrained': True,

    # ---------------- #
    # HYPER-PARAMETERS #
    # ---------------- #
    'epochs': 1,
    'epoch_to_resume': 0,

    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'clip_gradient': True,

    'learning_rate': 1e-4,
    'lr_momentum': 0.1,  # optimizer: SGD
    'lr_patience': 3,  # scheduler: ReduceLROnPlateau
    'lr_step_size': 3,  # scheduler: StepLR
    'lr_gamma': 0.1,  # scheduler: StepLR

    # ---- #
    # LOSS #
    # ---- #
    'alpha': 0.25,
    'gamma': 2.0,
    'lambda': 10,

}