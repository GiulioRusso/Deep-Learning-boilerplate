# 🧠 Deep Learning Boilerplate Project

This project is a deep learning boilerplate designed to facilitate the development, training, validation, and testing of neural networks. It provides a modular and well-structured approach for managing datasets, training models, computing metrics, and evaluating results.

## 🛠️ Installation steps
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd Deep-Learning-boilerplate
   ```
2. Create a virtual environment with `python3` or `conda` as you prefere:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
   ```sh
   conda create --name venv
   ```
3. Install dependencies based on your needs:
   ```sh
   pip install -r requirements.txt
   ```

## 🔩 Usage
### Running Training
To train a model, use the following command:
```sh
CUDA_VISIBLE_DEVICES=0 python3 -u main.py train --dataset=<dataset_name> --split=1-fold --norm=none --epochs=30 --lr=1e-04 --bs=2 --loss=FocalLoss > train.txt
```

### Running Testing
To test a model:
```sh
CUDA_VISIBLE_DEVICES=0 python3 -u main.py test --dataset=<dataset_name> --split=1-fold --norm=none --epochs=30 --lr=1e-04 --bs=2 --loss=FocalLoss > test.txt
```

### Resuming Training
If you want to resume a previously stopped training process:
```sh
CUDA_VISIBLE_DEVICES=0 python3 -u main.py resume --dataset=<dataset_name> --split=1-fold --norm=none --epochs=30 --lr=1e-04 --bs=2 --loss=FocalLoss > resume.txt
```

## Project Structure
```
Deep-Learning-boilerplate/
│── main.py                      # Entry point for training and evaluation
│── net/
│   ├── train.py                 # Training logic
│   ├── test.py                  # Testing logic
│   ├── validation.py            # Validation logic
│   ├── dataset/                 # Dataset management
│   ├── model/                   # Model definitions
│   ├── metrics/                 # Metric calculations (e.g., AUC, accuracy)
│   ├── optimizer/               # Optimizer configurations
│   ├── loss/                    # Loss functions
│   ├── scheduler/               # Learning rate schedulers
│   ├── initialization/          # Folder for setting up experiments
│   ├── classifications/         # Classification-related modules
│   ├── reproducibility/         # Reproducibility utilities (seed settings, etc.)
│   ├── resume/                  # Code for resuming training
│   └── plot/                    # Visualization utilities (e.g., loss curves, AUC plots)
```

## 🚨 Disclaimer
Please note that this project is not a fully operational model out of the box. Instead, it serves as a structured starting point that needs to be customized based on your dataset, task requirements (such as classification, detection, or segmentation), and model configurations.
For any questions or support, feel free to reach out. Contributions that improve usability are welcome! You are encouraged to fork the repository and submit your modifications.

