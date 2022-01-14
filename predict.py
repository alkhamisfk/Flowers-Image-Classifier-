import argparse
from PIL import Image
import torch 
from torchvision import datasets, transforms, models
from torch import nn, optim 
import torch.nn.functional as F
import numpy as np
import json

print('\n*--*--*--*--*--WELCOME--*--*--*--*--*--\n ')

parser = argparse.ArgumentParser(description='This is the prediction function to find your flower class')
parser.add_argument('--image_path', type=str, default='ImageClassifier/flowers/test/11/image_03141.jpg', required=True, help='full image path')
parser.add_argument('--checkpoint', type=str, default= 'checkpoint_vgg16.pth', required=True, help='load ur model checkpoint_vgg16.pth (the default) or checkpoint_alexnet.pth')
parser.add_argument('--top_k', type=int, default=5, required=True, help='top k value | default 5')
parser.add_argument('--category_names', type=str, default= 'ImageClassifier/cat_to_name.json', required=True, help='json file | default cat_to_name.json')
parser.add_argument('--gpu', default=True , required=True, help='GPU or CPU (for CPU enter False)')

args = parser.parse_args()
image_path = args.image_path
filepath = args.checkpoint
top_k = args.top_k
category_names = args.category_names

print('\nHere are your inputs for the Prediction:\n', '\nImage Path:', image_path, '\nProbabilities No:', top_k, '\njson file:', category_names,'GPU:',args.gpu) 

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

    
if filepath == 'checkpoint_alexnet.pth': 
    model = models.alexnet(pretrained=True)
    
elif filepath == 'checkpoint_vgg16.pth': 
    model = models.vgg16(pretrained=True)

    
# Load the Checkpoint-*--*--*--*--*--*--*--*--*--*-
print('\nLet\'s Load a Checkpoint !')

def load_checkpoint(file):
    
    # Trained on different source 
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    checkpoint_w = torch.load(file, map_location=map_location)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint_w['classifier']
    model.load_state_dict(checkpoint_w['state_dict'])
    model.class_to_idx = checkpoint_w['class_to_idx']
    model.eval()

    return model 



model = load_checkpoint(filepath)
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_im = Image.open(image_path)
    
    #resize, ANTIALIAS to reduce the distortion

    pil_im = pil_im.resize((256, 256), Image.ANTIALIAS)
    
    pil_im = pil_im.crop((16,16,240,240)) #crop

    np_image = np.array(pil_im) #convert
    
    np_image = np_image/ 255 #convert to range btw 0,1

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2, 0, 1))
    
    tensor_image = torch.FloatTensor(np_image)

    return tensor_image

print('\nLet\'s Process !')
process_image(image_path)

def predict(image_path, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to(device)

    pil_path = process_image(image_path)
    
    pil_path = pil_path.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(pil_path) 
        ps = torch.exp(output)
        probs, indices = ps.topk(topk, dim=1)

    probs, indices = probs.cpu(), indices.cpu()
    
    probs = np.array(probs).tolist()[0] 
    indices = np.array(indices).tolist()[0]

    idx_to_class={key:val for val,key in model.class_to_idx.items()}


    classes = [idx_to_class[x] for x in indices]

    return probs, classes

print('\nLet\'s Predict !')
probs, classes = predict(image_path, model, top_k)

def main():
    flowers_list = [cat_to_name[str(i)] for i in classes]  #str the index        
    print('\nThe predicted flowers\' classes with probabilities:')
    [print('P: {:.5f} Class: {}'.format(p, c.capitalize())) for p, c in zip(probs, flowers_list)]
    print('\n*--*--*--*--*--The Class of The Entered Flower--*--*--*--*--*\n \nImage:( {} )\nClass: {}\nProbability: {:.3f}'.format(image_path, flowers_list[0].upper(), probs[0]))
    print('\nPrediction is Completed!')

if __name__ == '__main__': 
    main()