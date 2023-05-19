#!/bin/bash

# Run the Python script with desired arguments

# Example 1: Plotting the loss over epochs for the MLP model on the SVHN dataset
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "MLP" --lr 0.05 --optimizer "SGD" --epochs 30

# Example 1: Plotting the loss over epochs for the CNN model on the SVHN dataset and different learning rates
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.01 --optimizer "SGD" --epochs 30
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.05 --optimizer "SGD" --epochs 30
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.1 --optimizer "SGD" --epochs 30
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.005 --optimizer "SGD" --epochs 30

# Example 2: Changing the optimizer to Adam and test different learning rates on CNN
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.001 --optimizer "Adam" --epochs 30
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.01 --optimizer "Adam" --epochs 30
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.1 --optimizer "Adam" --epochs 30

# Example 3: Changing the optimizer to RMSprop and test different learning rates on CNN
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.001 --optimizer "RMSprop" --epochs 30
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.01 --optimizer "RMSprop" --epochs 30
python group10_e03.py --plot_over "epochs" --dataloader "SVHN" --log-interval 300 --model "CNN" --lr 0.1 --optimizer "RMSprop" --epochs 30


