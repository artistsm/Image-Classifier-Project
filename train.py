#region Imports
import os
import argparse
#endregion

#region Global_Variables
sample_global_variable = None
#endregion

#region Classes
#create classes inside this region if any
#endregion

#region Argument_Parsing
#Define methods for argument types
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(path + " is not a valid path")
#Add arguments
parser = argparse.ArgumentParser(description="Train a new network on a dataset and optionalyy save the model as a checkpoint.")
parser.add_argument("data_dir", help="directory path containing training data", type=dir_path)
parser.add_argument("--save_dir", help="directory path to save checkpoints", type=str)
parser.add_argument("--arch", help="architecture", type=str, default="vgg13")
parser.add_argument("--learning_rate", help="learning rate for training", type=float, default=0.01)
parser.add_argument("--hidden_units", help="hidden units", type=int, default=512)
parser.add_argument("--epochs", help="number of epochs for training", type=int, default=20)
parser.add_argument("--gpu", help="use gpu for training", action="store_true")
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

args = parser.parse_args()      #Parse arguments
#endregion

#region Functions
def train():    #Main Train Function
    global args
    print (args)
    raise Exception("Function not implemented.")


#endregion

#region Main
if __name__ == "__main__":
    print args
    print ("Using data from path " + args.data_dir + " for training the model.")
    if args.gpu:
        print ("use gpu")
    if args.verbose:
        print ("print everything happening")
    if args.save_dir is not None:
        print ("do save checkpoints at every epoch inside folder " + args.save_dir)
        if not os.path.isdir(args.save_dir):    #if directory doesn't exist
            os.makedirs(args.save_dir)          #create directory
    train()
#endregion