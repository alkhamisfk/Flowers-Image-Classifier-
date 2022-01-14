import argparse
import torch 
from torchvision import datasets, transforms, models
from torch import nn, optim 
import torch.nn.functional as F
from time import time, sleep
import numpy as np

print('\n*--*--*--*--*--WELCOME--*--*--*--*--*--\n ')
parser = argparse.ArgumentParser(description='This is the training function to a model of ur choice ')
parser.add_argument('--data_dir',type=str, default='ImageClassifier/flowers', required=True, help='Write the directory')
parser.add_argument('--save_dir',type=str, default='checkpoint_vgg16.pth',  help='the directory to save the checkpoint suggestion: checkpoint_ARCH.pth replace ARCH with vgg16(default) or alexnet | None')
parser.add_argument('--arch',type=str, default='vgg16', required=True, help='model (vgg16 or alexnet) | default vgg16')
parser.add_argument('--hidden_units', type=int, default=512, required=True, help='hidden layers number | default 512')
parser.add_argument('--learning_rate', type=float, default=0.001, required=True, help='learning rate value | default 0.001')
parser.add_argument('--epochs', type=int, default=5, required=True, help='epochs value | default 5')
parser.add_argument('--gpu', action='store', default=True, required=True, help='True: GPU or False: CPU')

args = parser.parse_args()


arch = args.arch
hidden_units = args.hidden_units
learning_rate = args.learning_rate
epochs = args.epochs
data_dir = args.data_dir
m= args.save_dir

print('\nHere are your inputs for the Training:\n', '\nDirectory:', data_dir, '\nModel Arch:', arch, '\nNo. of Hidden Layers:', hidden_units, 'Learning Rate:', learning_rate, 'Epochs:', epochs, 'GPU',args.gpu) 

print('\nLet\'s Start! ')

# Directory-*--*--*--*--*--*--*--*--*--*-
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Data Transforms-*--*--*--*--*--*--*--*--*--*-
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229,0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229,0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229,0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

print('\nLet\'s Build the Network!')


# Build the Network-*--*--*--*--*--*--*--*--*--*-
#if args.gpu == True: 
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")


if arch == 'alexnet': 
    model = models.alexnet(pretrained=True)
    inputs = 9216
    
elif arch == 'vgg16': 
    model = models.vgg16(pretrained=True)
    inputs = 25088

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(inputs, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units,102),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


model.to(device);
model

print('\nLet\'s Train the Network!')
print('\nPlease Wait......')

# Train the Network-*--*--*--*--*--*--*--*--*--*-
start_time = time()
steps = 0
running_loss = 0 
print_every = 64
train_losses, valid_losses = [], []
for e in range(epochs):
    for images, labels in trainloader:
        steps+= 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            valid_loss = 0 
            accuracy = 0 
            model.eval()
            with torch.no_grad():
                for images_val, labels_val in validloader:
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    output = model.forward(images_val)
                    batch_loss = criterion(output, labels_val)
                    valid_loss += batch_loss.item()
                    
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels_val.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            train_losses.append(running_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))
            percentage=accuracy/len(validloader)
    
            print("Epoch: {}/{}".format(e+1,epochs),          
                  " Training Loss: {:.3f}".format(running_loss/len(trainloader)), 
                  " Validation Loss {:.3f}".format(valid_loss/len(validloader)),
                  " Accuracy: {:.3f}".format(100*percentage))
        
            running_loss = 0 
            model.train()
               
end_time = time()
total_time = end_time - start_time 
print("\n** Total Elapsed Runtime:",
      str(int((total_time/3600)))+":"+str(int((total_time%3600)/60))+":"
      +str(int((total_time%3600)%60))) 
  
# Save the Checkpoint-*--*--*--*--*--*--*--*--*--*-

model.class_to_idx = train_data.class_to_idx
checkpoint_w = {'input_size': inputs,
                'output_size': 102,
                'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'optimizer_state': optimizer.state_dict}

#m = 'checkpoint_' + arch +'.pth' #in case you want to remove save_dir as input


torch.save(checkpoint_w, m)
print('\nCheckpoint is SAVED!')

def main():
    print('\nTraining is Completed!')

if __name__ == '__main__':
    main()