#region Imports
import os
import argparse
#endregion

#region Global_Variables
sample_global_variable = None
img_file_formats = ("img","bmp","jpg","jpeg","png")
#endregion

#region Classes
#Create classes if any inside this region
#endregion

#region Argument_Parsing
#Define methods for argument types
def img_path(path):
    global img_file_formats
    ext = os.path.splitext(path)[1][1:]
    if os.path.isfile(path) and ext.lower() in img_file_formats:
        return path
    else:
        raise argparse.ArgumentTypeError(path + " is not a valid image file. Allowed file formats are "+img_file_formats)

def json_file(path):
    ext = os.path.splitext(path)[1][1:]
    if os.path.isfile(path) and ext.lower() == "json":
        return path
    else:
        raise argparse.ArgumentTypeError(path + " is not a valid json file.")

#Add arguments
parser = argparse.ArgumentParser(description="Predict class name from an image along with the probability of that name.")
parser.add_argument("image_path", help="path to image file", type=img_path)
parser.add_argument("checkpoint", help="checkpoint", type=str)  #don't know what type this should be
parser.add_argument("--top_k", metavar="K", help="return top k most likely classes", type=int, default=5)
parser.add_argument("--category_names", metavar="category_names.json", help="path to json file mapping category to real names", type=json_file)
parser.add_argument("--gpu", help="use gpu for training", action="store_true")
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

args = parser.parse_args()      #Parse arguments
#endregion

#region Functions
def parse_json_file(filename):  #return data from json file
    import json
    with open('cat_to_name.json', 'r') as f:
        json_data = json.load(f)
    return json_data

def run_prediction(): #Run main prediction process
    global args
    print (args)    #print all arguments
    if args.gpu:
        print ("Using gpu.")

    json_data = parse_json_file(args.category_names) #get json data
    print (json_data)

    import torch
    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from torch import nn
    from torch import optim
    from torchvision import datasets, transforms, models
    from collections import OrderedDict

    # TODO: Write a function that loads a checkpoint and rebuilds the model
    # Write a function that loads a checkpoint and rebuilds the model
    def load_checkpoint():
        """
        Loads deep learning model checkpoint.
        """
        
        # Load the saved file
        checkpoint = torch.load(args.checkpoint)
        
        # Download pretrained model
        model = models.vgg16(pretrained=True)
        
        # Freeze parameters so we don't backprop through them
        for param in model.parameters(): param.requires_grad = False
        
        # Load stuff from checkpoint
        model.class_to_idx = checkpoint['class_to_idx']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])

        
        return model

    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        test_image = PIL.Image.open(image)

        # Get original dimensions
        orig_width, orig_height = test_image.size

        # Find shorter size and create settings to crop shortest side to 256
        if orig_width < orig_height: resize_size=[256, 256**600]
        else: resize_size=[256**600, 256]
            
        test_image.thumbnail(size=resize_size)

        # Find pixels to crop on to create 224x224 image
        center = orig_width/4, orig_height/4
        left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
        test_image = test_image.crop((left, top, right, bottom))

        # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
        np_image = np.array(test_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

        # Normalize each color channel
        normalise_means = [0.485, 0.456, 0.406]
        normalise_std = [0.229, 0.224, 0.225]
        np_image = (np_image-normalise_means)/normalise_std
            
        # Set the color to the first channel
        np_image = np_image.transpose(2, 0, 1)
        
        return np_image
        
    def predict(image_path, model, top_k=args.top_k):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        
        image_path: string. Path to image, directly to image and not to folder.
        model: pytorch neural network.
        top_k: integer. The top K classes to be calculated
        
        returns top_probabilities(k), top_labels
        '''
        
        # No need for GPU on this part (just causes problems)
        model.to("cpu")
        if args.gpu:
            model.to("gpu")
        
        
        # Set model to evaluate
        model.eval()

        # Convert image from numpy to torch
        torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                    axis=0)).type(torch.FloatTensor).to("cpu")

        # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
        log_probs = model.forward(torch_image)

        # Convert to linear scale
        linear_probs = torch.exp(log_probs)

        # Find the top 5 results
        top_probs, top_labels = linear_probs.topk(top_k)
        
        # Detatch all of the details
        top_probs = np.array(top_probs.detach())[0] # This is not the correct way to do it but the correct way isnt working thanks to cpu/gpu issues so I don't care.
        top_labels = np.array(top_labels.detach())[0]
        
        # Convert to classes
        idx_to_class = {val: key for key, val in    
                                        model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_labels]
        top_flowers = [cat_to_name[lab] for lab in top_labels]
        
        return top_probs, top_labels, top_flowers
    
    flower_num = args.image_path.split('/')[2]
    cat_to_name = parse_json_file(args.category_names)
    title_ = cat_to_name[flower_num]
    img = process_image(args.image_path)
    model = load_checkpoint()
    probs, labs, flowers = predict(args.image_path, model) 
    print (probs, labs, flowers)


#endregion

#region Main
if __name__ == "__main__":
    run_prediction()

#endregion