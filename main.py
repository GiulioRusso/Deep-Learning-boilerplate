import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from pandas import read_csv
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from net.parameters.parameters import parameters_parsing
from net.initialization.ID.experimentID import experimentID
from net.reproducibility.reproducibility import reproducibility
from net.dataset.MyDataset import MyDataset
from net.dataset.dataset_split import dataset_split
from net.initialization.init import initialization
from net.dataset.utility.read_split import read_split
from net.dataset.dataset_transforms import dataset_transforms
from net.model.MyNetwork import MyNetwork
from net.loss.get_loss import get_loss
from net.optimizer.get_optimizer import get_optimizer
from net.scheduler.get_scheduler import get_scheduler
from net.initialization.dict.metrics import metrics_dict
from net.evaluation.current_learning_rate import current_learning_rate
from net.metrics.metrics_train import metrics_train_csv
from net.metrics.show_metrics.show_metrics_train import show_metrics_train
from net.train import train
from net.validation import validation
from net.evaluation.ROC_AUC import ROC_AUC
from net.metrics.utility.my_notation import scientific_notation
from net.plot.ROC_AUC_plot import ROC_AUC_plot
from net.plot.loss_plot import loss_plot
from net.plot.utility.figure_size import figure_size
from net.model.utility.save_model import save_best_model, save_resume_model
from net.resume.resume import resume
from net.test import test
from net.metrics.metrics_test import metrics_test_csv
from net.metrics.show_metrics.show_metrics_test import show_metrics_test
from net.model.utility.load_model import load_best_model


def main():
    print("| ============================ |\n"
          "|   MY DEEP LEARNING PROJECT   |\n"
          "| ============================ |\n")

    # ================== #
    # PARAMETERS-PARSING #
    # ================== #
    # command line parameter parsing
    parser = parameters_parsing()

    # ============== #
    # INITIALIZATION #
    # ============== #
    print("\n---------------"
          "\nINITIALIZATION:"
          "\n---------------")

    # experiment ID
    experiment_ID, experiment_resume_ID = experimentID(parser=parser)

    # path initialization
    path = initialization(network_name="MyNetwork",
                          experiment_ID=experiment_ID,
                          experiment_resume_ID=experiment_resume_ID,
                          parser=parser)

    # read split
    data_split = read_split(path_split_case=path['dataset']['split'])

    # ====== #
    # DEVICE #
    # ====== #
    print("\n-------"
          "\nDEVICE:"
          "\n-------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(parser.num_threads)
    print("GPU device name: {}".format(torch.cuda.get_device_name(0)))

    # =============== #
    # REPRODUCIBILITY #
    # =============== #
    reproducibility(seed=parser.seed)

    # ============ #
    # LOAD DATASET #
    # ============ #
    print("\n-------------"
          "\nLOAD DATASET:"
          "\n-------------")

    # load dataset
    dataset = MyDataset(images_dir=path['dataset']['images'],
                        annotations_dir=path['dataset']['annotations'],
                        filename_list=path['dataset']['list'],
                        transforms=None)

    # TODO: define number of classes based on your task

    num_classes = parser.num_classes

    # ============= #
    # DATASET SPLIT #
    # ============= #
    # subset dataset according to data split
    dataset_train, dataset_val, dataset_test = dataset_split(data_split=data_split,
                                                             dataset=dataset)

    # ================== #
    # DATASET TRANSFORMS #
    # ================== #
    # original view
    train_transforms, val_transforms, test_transforms = dataset_transforms(parser=parser,
                                                                           normalization=parser.norm,
                                                                           statistics_path=path['dataset']['statistics'])

    # apply dataset transforms
    dataset_train.dataset.transforms = train_transforms
    dataset_val.dataset.transforms = val_transforms
    dataset_test.dataset.transforms = test_transforms

    # ============ #
    # DATA LOADERS #
    # ============ #
    # dataloader-train
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=parser.batch_size_train,
                                  shuffle=True,
                                  num_workers=parser.num_workers,
                                  pin_memory=True)

    # dataloader-val
    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=parser.batch_size_val,
                                shuffle=False,
                                num_workers=parser.num_workers,
                                pin_memory=True)

    # dataloader-test
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=parser.batch_size_test,
                                 shuffle=False,
                                 num_workers=parser.num_workers,
                                 pin_memory=True)

    # ============= #
    # NETWORK MODEL #
    # ============= #
    # net
    net = MyNetwork()

    # data parallel
    net = nn.DataParallel(module=net)

    # move net to device
    net.to(device)

    # =========== #
    # MODE: TRAIN #
    # =========== #
    if parser.mode in ['train', 'resume']:

        # ========= #
        # OPTIMIZER #
        # ========= #
        optimizer = get_optimizer(net_parameters=net.parameters(),
                                  parser=parser)

        # ========= #
        # SCHEDULER #
        # ========= #
        scheduler = get_scheduler(optimizer=optimizer,
                                  parser=parser)

        # ========= #
        # CRITERION #
        # ========= #
        criterion = get_loss(loss=parser.loss,
                             device=device,
                             parser=parser)

        # ==================== #
        # INIT METRICS (TRAIN) #
        # ==================== #
        metrics = metrics_dict(metrics_type='train')

        # training epochs range
        start_epoch_train = 1  # star train
        stop_epoch_train = start_epoch_train + parser.epochs  # stop train

        # ============ #
        # MODE: RESUME #
        # ============ #
        if parser.mode in ['resume']:
            # --------------- #
            # RESUME TRAINING #
            # --------------- #
            start_epoch_train, stop_epoch_train = resume(experiment_ID=experiment_ID,
                                                         net=net,
                                                         optimizer=optimizer,
                                                         scheduler=scheduler,
                                                         path=path,
                                                         parser=parser)

        # for each epoch
        for epoch in range(start_epoch_train, stop_epoch_train):

            # ======== #
            # TRAINING #
            # ======== #
            print("\n---------"
                  "\nTRAINING:"
                  "\n---------")
            time_train_start = time.time()

            loss = train(num_epoch=epoch,
                         epochs=parser.epochs,
                         net=net,
                         num_classes=num_classes,
                         dataloader=dataloader_train,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         criterion=criterion,
                         device=device,
                         parser=parser)

            time_train = time.time() - time_train_start

            # ========== #
            # VALIDATION #
            # ========== #
            print("\n-----------"
                  "\nVALIDATION:"
                  "\n-----------")
            time_val_start = time.time()

            validation(num_epoch=epoch,
                       epochs=parser.epochs,
                       net=net,
                       dataloader=dataloader_val,
                       classifications_path=path['classifications']['validation'],
                       device=device)

            time_val = time.time() - time_val_start

            # ==================== #
            # METRICS (VALIDATION) #
            # ==================== #
            time_metrics_val_start = time.time()

            # TODO: modify the header based on your needs

            # read classifications validation for evaluation (numpy array)
            classifications_header = ["FILENAME", "PREDICTION", "SCORE", "ANNOTATION"]
            classifications_validation = read_csv(filepath_or_buffer=path['classifications']['validation'], usecols=classifications_header).values

            # TODO: evaluate the metrics you need

            # compute ROC AUC
            ROC_AUC_val = ROC_AUC(classifications=classifications_validation)

            # =============== #
            # PLOT VALIDATION #
            # =============== #
            print("\n----------------"
                  "\nPLOT VALIDATION:"
                  "\n----------------")

            # get current learning rate
            last_learning_rate = current_learning_rate(scheduler=scheduler,
                                                       optimizer=optimizer,
                                                       parser=parser)

            time_metrics_val = time.time() - time_metrics_val_start

            # TODO: keep track of your metrics

            # update performance
            metrics['ticks'].append(epoch)
            metrics['loss'].append(loss)
            metrics['learning_rate'].append(scientific_notation(number=last_learning_rate))
            metrics['ROC_AUC'].append(ROC_AUC_val)
            metrics['time']['train'].append(time_train)
            metrics['time']['validation'].append(time_val)
            metrics['time']['metrics'].append(time_metrics_val)

            # TODO: update metrics file and print

            # metrics-train.csv
            metrics_train_csv(metrics_path=path['metrics']['train'],
                              metrics=metrics)

            # show metrics train
            show_metrics_train(metrics=metrics)

            # =============== #
            # SAVE BEST MODEL #
            # =============== #
            print("\n----------------"
                  "\nSAVE BEST MODEL:"
                  "\n----------------")

            # TODO: save the model based on your metrics
            # save best-model with ROC AUC
            if (epoch - 1) == np.argmax(metrics['ROC_AUC']):
                save_best_model(epoch=epoch,
                                net=net,
                                metrics=metrics['ROC_AUC'],
                                metrics_type='ROC AUC',
                                optimizer=optimizer,
                                scheduler=scheduler,
                                path=path['models']['best'])

            # save resume-model
            save_resume_model(epoch=epoch,
                              net=net,
                              ROC_AUC=metrics['ROC_AUC'][-1],
                              optimizer=optimizer,
                              scheduler=scheduler,
                              path=path['models']['resume'])

            # ========== #
            # PLOT TRAIN #
            # ========== #
            print("\n-----------"
                  "\nPLOT TRAIN:"
                  "\n-----------")

            # figure size
            figsize_x, figsize_y = figure_size(epochs=parser.epochs)

            # epochs ticks
            epochs_ticks = np.arange(1, parser.epochs + 1, step=1)

            # loss plot
            loss_plot(figsize=(figsize_x, figsize_y),
                      title="LOSS",
                      experiment_ID=experiment_ID,
                      ticks=metrics['ticks'],
                      epochs_ticks=epochs_ticks,
                      loss=metrics['loss'],
                      loss_path=path['plots']['train']['loss'])

            # ROC AUC plot
            ROC_AUC_plot(figsize=(figsize_x, figsize_y),
                         title="ROC AUC",
                         experiment_ID=experiment_ID,
                         ticks=metrics['ticks'],
                         epochs_ticks=epochs_ticks,
                         ROC_AUC=metrics['ROC_AUC'],
                         ROC_AUC_path=path['plots']['validation']['ROC_AUC'])

        # ========== #
        # MODE: TEST #
        # ========== #
        if parser.mode in ['test']:
            # =================== #
            # INIT METRICS (TEST) #
            # =================== #
            metrics = metrics_dict(metrics_type='test')

            # =============== #
            # LOAD BEST MODEL #
            # =============== #
            print("\n----------------"
                  "\nLOAD BEST MODEL:"
                  "\n----------------")

            # load best model ROC AUC
            if parser.load_best_ROC_AUC_model:
                load_best_model(net=net,
                                metrics_type='accuracy',
                                path=path['models']['best'])

            # ==== #
            # TEST #
            # ==== #
            print("\n-----"
                  "\nTEST:"
                  "\n-----")
            time_test_start = time.time()
            test(net=net,
                 dataloader=dataloader_test,
                 classifications_path=path['classifications']['test'],
                 device=device)
            time_test = time.time() - time_test_start

            # ============== #
            # METRICS (TEST) #
            # ============== #
            time_metrics_test_start = time.time()

            # read classifications test for evaluation (numpy array)
            classifications_header = ["FILENAME", "PREDICTION", "SCORE", "ANNOTATION"]
            classifications_test = read_csv(filepath_or_buffer=path['classifications']['test'], usecols=classifications_header).values

            # compute ROC AUC
            ROC_AUC_test = ROC_AUC(classifications=classifications_test)

            time_metrics_test = time.time() - time_metrics_test_start

            # update performance
            metrics['ROC_AUC'].append(ROC_AUC_test)
            metrics['time']['test'].append(time_test)
            metrics['time']['metrics'].append(time_metrics_test)

            # metrics-test.csv
            metrics_test_csv(metrics_path=path['metrics']['test'],
                             metrics=metrics)

            # show
            show_metrics_test(metrics=metrics)


if __name__ == "__main__":
    main()
