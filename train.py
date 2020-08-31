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
parser.add_argument("--learning_rate", help="learning rate for training", type=float, default=0.001)
parser.add_argument("--hidden_units", help="hidden units", type=int, default=2048)
parser.add_argument("--epochs", help="number of epochs for training", type=int, default=10)
parser.add_argument("--gpu", help="use gpu for training", action="store_true")
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

args = parser.parse_args()      #Parse arguments
#endregion

#region Functions
def train():    #Main Train Function
    global args
    
    # Imports here
    import torch
    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from torch import nn
    from torch import optim
    from torchvision import datasets, transforms, models
    from collections import OrderedDict
    import importlib

    # %matplotlib inline
    # %config InlineBackend.figure_format = 'retina'
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose ([transforms.RandomRotation (30),
                                                 transforms.RandomResizedCrop (224),
                                                 transforms.RandomHorizontalFlip (),
                                                 transforms.ToTensor (),
                                                 transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    valid_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                 transforms.CenterCrop (224),
                                                 transforms.ToTensor (),
                                                 transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    test_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                 transforms.CenterCrop (224),
                                                 transforms.ToTensor (),
                                                 transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ]) 

    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder (train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder (valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder (test_dir, transform = test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=64, shuffle = True)

    image, labels = next(iter(trainloader))
    image.shape
    train_image_datasets.class_to_idx
    # TODO: Build and train your network
    if args.arch == "resnet18":
        model = models.resnet18(pretrained = True)
    elif args.arch == "alexnet":
        model = models.alexnet(pretrained = True)
    elif args.arch == "vgg16":
        model= models.vgg16(pretrained = True)
    elif args.arch == "squeezenet1_0":
        model= models.squeezenet1_0(pretrained = True)
    elif args.arch == "densenet161":
        model= models.densenet161(pretrained = True)
    elif args.arch == "inception_v3":
        model= models.inception_v3(pretrained = True)
    elif args.arch == "googlenet":
        model= models.googlenet(pretrained = True)
    elif args.arch == "shufflenet_v2_x1_0":
        model= models.shufflenet_v2_x1_0(pretrained = True)
    elif args.arch == "mobilenet_v2":
        model= models.mobilenet_v2(pretrained = True)
    elif args.arch == "resnext50_32x4d":
        model= models.resnext50_32x4d(pretrained = True)
    elif args.arch == "wide_resnet50_2":
        model= models.wide_resnet50_2(pretrained = True)
    elif args.arch == "mnasnet1_0":
        model= models.mnasnet1_0(pretrained = True)
    else:
        print("ERROR: Invalid architecture!!")

    # updating Classifer in the network according to our 
    for param in model.parameters(): 
        param.requires_grad = False

    classifier = nn.Sequential  (OrderedDict ([
                                ('fc1', nn.Linear (9216, 4096)),
                                ('relu1', nn.ReLU ()),
                                ('dropout1', nn.Dropout (p = 0.3)),
                                ('fc2', nn.Linear (4096, args.hidden_units)),
                                ('relu2', nn.ReLU ()),
                                ('dropout2', nn.Dropout (p = 0.3)),
                                ('fc3', nn.Linear (args.hidden_units, 102)),
                                ('output', nn.LogSoftmax (dim =1))
                                ]))
    model.classifier = classifier

    #initializing criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters (), lr = args.learning_rate)

    # Device agnostic code, automatically uses CUDA if it's enabled
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    device


    # change to device
    model.to(device)

    # Using a validation function
    def validation(model, testloader, criterion):
        model.to(device)
        
        test_loss = 0
        accuracy = 0
        
        for image,labels in testloader:
            
            image, labels = image.to(device), labels.to(device)
            
            output = model.forward(image)
            test_loss += criterion(output, labels).item()
            
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        return test_loss, accuracy

    print(device)

    #Training the classifer using our data

    epochs = args.epochs
    print_every = 40
    steps = 0
    #initializing criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters (), lr = args.learning_rate)

    for e in range(epochs):
        running_loss = 0
        model.train() 
        
        for image, labels in trainloader:
            steps += 1
            
            image, labels = image.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
                
                print("Epoch: {}/{} | ".format(e+1, epochs),
                    "Training Loss: {:.4f} | ".format(running_loss/print_every),
                    "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                    "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                model.train()

    # TODO: Save the checkpoint 
    # Create this `class_to_idx` attribute quickly
    model.class_to_idx = train_image_datasets.class_to_idx

    checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'mapping':    model.class_to_idx
             }  

    torch.save(checkpoint, 'my_checkpoint.pth')
    print("Training Completed!! model saved as my_checkpoint.pth")

#endregion

#region Main
if __name__ == "__main__":
    print (args)
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
    print ("Training Complete.")
#endregion