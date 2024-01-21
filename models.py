import torchvision
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader, Dataset # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder
from PIL import Image
import torch

device = torch.device("cuda")
trans=[transforms.RandomHorizontalFlip(p=1),transforms.RandomVerticalFlip(p=1),transforms.RandomRotation(90)]
trans2=transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)


class ImageLoaderRAM(Dataset):
    def __init__(self, dataset, transform,train):
        self.transform = transform
        self.dataset = self.checkChannel(dataset) 

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        image =self.dataset[item][0]
        classCategory = self.dataset[item][1]
        return image, classCategory

    def checkChannel(self, dataset):
        datasetRGB = []
        for index in tqdm(range(len(dataset))):
            img=Image.open(dataset[index][0])
            if (img.getbands() == ("R", "G", "B")): # Check Channels
                img=self.transform(img)
                label=dataset[index][1]
                datasetRGB.append((img,label))
        return datasetRGB
    
class ImageLoader(Dataset):
    def __init__(self, dataset, transform):
        self.transform = transform
        self.dataset = self.checkChannel(dataset) # some images are CMYK, Grayscale, check only RGB 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        image =self.transform(Image.open(self.dataset[item][0]))
        classCategory = self.dataset[item][1]
        
        return image, classCategory
        
    
    def checkChannel(self, dataset):
        datasetRGB = []
        for index in tqdm(range(len(dataset))):
            if (Image.open(dataset[index][0]).getbands() == ("R", "G", "B")): # Check Channels
                datasetRGB.append(dataset[index])
        return datasetRGB
    
class AllImageLoader(Dataset):
    def __init__(self, dataset,detail_label, transform):
        self.transform = transform
        self.imgs,self.labels = self.checkChannel(dataset) # some images are CMYK, Grayscale, check only RGB 
        self.detail_label = detail_label
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, item):
        image =self.imgs[item]
        classCategory = self.labels[item]
        detail_label = self.detail_label[item]
        return image, classCategory, detail_label
        
    def get_detail_label(self,img):
        index=0
        for i in range(len(self.imgs)):
            if torch.equal(img,self.imgs[i]):
                index=i
        return self.detail_label[index]

    def checkChannel(self, dataset):
        datasetRGB = []
        for index in tqdm(range(len(dataset))):
            if (Image.open(dataset[index][0]).getbands() == ("R", "G", "B")): # Check Channels
                datasetRGB.append(dataset[index])
        imgs=[]
        labels=[]
        for img_dir,label in tqdm(datasetRGB):
            img=self.transform(Image.open(img_dir)).to('cuda')
            imgs.append(img)
            labels.append(label)
        return imgs,labels






from tqdm import tqdm
from torchvision import models
import os
# load pretrain model and modify...
def get_resnet50(classes,name,pretrained=False):
    model = models.resnet50(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    if pretrained and os.path.exists(f'{name}.pt'):
        print(f'{name}.pt exists')
        model.load_state_dict(torch.load(f'{name}.pt'))
        return model.to(device)
    model.to(device)
    return model

def get_resnet152(classes,name,pretrained=False):
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    if pretrained and os.path.exists(f'{name}.pt'):
        print(f'{name}.pt exists')
        model.load_state_dict(torch.load(f'{name}.pt'))
        return model.to(device)
    model.to(device)
    return model

def get_densenet121(classes,name,pretrained=False):
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, classes)
    if pretrained and os.path.exists(f'{name}.pt'):
        print(f'{name}.pt exists')
        model.load_state_dict(torch.load(f'{name}.pt'))
        return model.to(device)
    model.to(device)
    return model

def get_densenet169(classes,name,pretrained=False):
    model = models.densenet169(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, classes)
    if pretrained and os.path.exists(f'{name}.pt'):
        print(f'{name}.pt exists')
        model.load_state_dict(torch.load(f'{name}.pt'))
        return model.to(device)
    model.to(device)
    return model

def get_alexnet(classes,name,pretrained=False):
    model = models.alexnet(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, classes)
    if pretrained and os.path.exists(f'{name}.pt'):
        print(f'{name}.pt exists')
        model.load_state_dict(torch.load(f'{name}.pt'))
        return model.to(device)
    model.to(device)
    return model

def get_vgg16(classes,name,pretrained=False):
    model = models.vgg16(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, classes)
    if pretrained and os.path.exists(f'{name}.pt'):
        print(f'{name}.pt exists')
        model.load_state_dict(torch.load(f'{name}.pt'))
        return model.to(device)
    model.to(device)
    return model

def get_vgg11(classes,name,pretrained=False):
    model = models.vgg11(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, classes)
    if pretrained and os.path.exists(f'{name}.pt'):
        print(f'{name}.pt exists')
        model.load_state_dict(torch.load(f'{name}.pt'))
        return model.to(device)
    model.to(device)
    return model

def getmodel(classes,name,model_type,pretrained=False):
    model=None
    if model_type=='resnet50':
        model=get_resnet50(classes,name,pretrained)
    elif model_type=='resnet152':
        model=get_resnet152(classes,name,pretrained)
    elif model_type=='vgg16':
        model=get_vgg16(classes,name,pretrained)
    elif model_type=='vgg11':
        model=get_vgg11(classes,name,pretrained)
    elif model_type=='densenet121':
        model=get_densenet121(classes,name,pretrained)
    elif model_type=='densenet169':
        model=get_densenet169(classes,name,pretrained)
    elif model_type=='alexnet':
        model=get_alexnet(classes,name,pretrained)
    else:
        raise ValueError('Invalid model type')
    return model