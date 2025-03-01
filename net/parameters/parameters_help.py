from net.parameters.parameters_default import parameters_default

parameters_help = {

    # -------------- #
    # EXECUTION MODE #
    # -------------- #
    'mode': f"execution mode (train, resume, test)",

    # ------- #
    # DATASET #
    # ------- #
    'dataset': f"dataset name (default: '{parameters_default['dataset']}')",
    'num_classes': f"number of classes (default: {parameters_default['num_classes']})",
    'image_height': f"image height size (default: {parameters_default['image_height']}",
    'image_width': f"image width size (default: {parameters_default['image_width']}",
    'split': f"dataset split (default: '{parameters_default['split']}')",
    'norm': f"dataset normalization (default: {parameters_default['norm']}",
    'do_dataset_augmentation': f"do dataset augmentation (default: {parameters_default['do_dataset_augmentation']}",

    # ------ #
    # DEVICE #
    # ------ #
    'GPU': f"GPU device name (default: {parameters_default['GPU']}",
    'num_threads': f"number of threads (default: {parameters_default['num_threads']}",

    # --------------- #
    # REPRODUCIBILITY #
    # --------------- #
    'seed': f"seed for reproducibility (default: {parameters_default['seed']})",

    # ----------- #
    # DATA LOADER #
    # ----------- #
    'batch_size_train': f"batch size for train (default: {parameters_default['batch_size_train']})",
    'batch_size_val': f"batch size for validation (default: {parameters_default['batch_size_val']})",
    'batch_size_test': f"batch size for test (default: {parameters_default['batch_size_test']})",

    'num_workers': f"numbers of sub-processes to use for data loading, if 0 the data will be loaded in the main process (default: {parameters_default['num_workers']})",

    # ------- #
    # NETWORK #
    # ------- #
    'network_name': f"Network name (default: {parameters_default['network_name']})",
    'backbone': f"Backbone model (default: {parameters_default['backbone']})",
    'pretrained': f"PreTrained model (default: {parameters_default['pretrained']})",

    # ---------------- #
    # HYPER-PARAMETERS #
    # ---------------- #
    'epochs': f"number of epochs (default: {parameters_default['epochs']})",
    'epoch_to_resume': f"number of epoch to resume (default: {parameters_default['epoch_to_resume']})",

    'optimizer': f"Optimizer (default: '{parameters_default['optimizer']}'",
    'scheduler': f"Scheduler (default: '{parameters_default['scheduler']}'",
    'clip_gradient': f"Clip Gradient (default: '{parameters_default['clip_gradient']}'",

    'learning_rate': f"how fast approach the minimum (default: {parameters_default['learning_rate']})",
    'lr_momentum': f"momentum factor [optimizer: SGD] (default: {parameters_default['lr_momentum']})",
    'lr_patience': f"number of epochs with no improvement after which learning rate will be reduced [scheduler: ReduceLROnPlateau] (default: {parameters_default['lr_patience']})",
    'lr_step_size': f"how much the learning rate decreases [scheduler: StepLR] (default: {parameters_default['lr_step_size']})",
    'lr_gamma': f"multiplicative factor of learning rate decay [scheduler: StepLR] (default: {parameters_default['lr_gamma']})",

    # ---- #
    # LOSS #
    # ---- #
    'alpha': f"alpha parameter for loss (default: {parameters_default['alpha']}",
    'gamma': f"gamma parameter for loss (default: {parameters_default['gamma']}",
    'lambda': f"lambda factor for loss sum (default: {parameters_default['lambda']}",

}