import os
import argparse

from kfold_cross_validation import kfold_main
from hold_out_split_validation import holdout_main

def get_args():
    parser = argparse.ArgumentParser()

    # model-infos
    #######################################################################################
    parser.add_argument("--run_name", type=str, default="test_default",
                        help="the name assigned to the current run")

    parser.add_argument("--model_name", type=str, default="binaryclassif_img_model",
                        help="the name of the model to be saved or loaded")
    #######################################################################################

    # training-parameters (1)
    #######################################################################################
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="the total number of training epochs")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="the batch size for training and validation data")

    # https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/szymon_migacz-pytorch-performance-tuning-guide.pdf    
    parser.add_argument("--workers", type=int, default=4,
                        help="the number of workers in the data loader")
    #######################################################################################

    # training-parameters (2)
    #######################################################################################
    parser.add_argument("--lr", type=float, default=0.001,
                        help="the learning rate for optimization")

    parser.add_argument("--loss", type=str, default="bcewll",
                        choices=["bcewll"],
                        help="the loss function used for model optimization")

    parser.add_argument("--opt", type=str, default="Adam", 
                        choices=["SGD", "Adam", "RMSprop"],
                        help="the optimizer used for training")

    parser.add_argument("--patience", type=int, default=5,
                        help="the threshold for early stopping during training")
    #######################################################################################

    # training-parameters (3)
    #######################################################################################
    parser.add_argument("--load_model", action="store_true",
                        help="determines whether to load the model from a checkpoint")

    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", 
                        help="the path to save the trained model")

    parser.add_argument("--num_classes", type=int, default=1,
                        help="the number of classes to predict with the final Linear layer")
    #######################################################################################

    # data-path
    #######################################################################################
    parser.add_argument("--raw_data_path", type=str, default="./data/Images",
                        help="path where to get the raw-dataset")
    #######################################################################################

    # data transformation
    #######################################################################################
    parser.add_argument("--apply_transformations", action="store_true", default=True,
                        help="indicates whether to apply transformations to images")
    #######################################################################################

    # # Google Colab
    # #######################################################################################
    # parser.add_argument("-f", "--file", required=False)
    # #######################################################################################

    return parser.parse_args()

# check if the script is being run as the main program
if __name__ == "__main__":
    # parse command line arguments
    args = get_args()
    
    # if folder doesn't exist, then create it
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.isdir("statistics"):
        os.makedirs("statistics")
    
    print(f"\n{args}")
    # kfold_main(args)
    holdout_main(args)
