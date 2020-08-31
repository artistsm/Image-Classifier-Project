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
parser.add_argument("--top_k", metavar="K", help="return top k most likely classes", type=int, default=3)
parser.add_argument("--category_names", metavar="category_names.json", help="path to json file mapping category to real names", type=json_file)
parser.add_argument("--gpu", help="use gpu for training", action="store_true")
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

args = parser.parse_args()      #Parse arguments
#endregion

#region Functions
def parse_json_file(filename):  #return data from json file
    import json
    with open(filename, 'r') as myfile:
        data=myfile.read()
    obj = json.loads(data)
    return obj

def predict(): #Run main prediction process
    global args
    print (args)    #print all arguments
    if args.gpu:
        print ("Using gpu.")

    json_data = parse_json_file(args.category_names) #get json data
    print (json_data)

    raise Exception("Function not implemented.")


#endregion

#region Main
if __name__ == "__main__":
    predict()

#endregion