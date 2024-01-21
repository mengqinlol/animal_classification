import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import models
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.

from tqdm import tqdm

device = torch.device("cuda") # use gpu or cpu
bs=32
from sklearn.model_selection import train_test_split

# print(train_data,train_label)
# ImageLoader Class

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
]) # train transform

test_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
]) # test transform

model_type="resnet50"
# Loss and optimizer
model=models.getmodel(3,'first',model_type)
cat_model=models.getmodel(12,'cat',model_type)
dog_model=models.getmodel(25,'dog',model_type)
bird_model=models.getmodel(10,'bird',model_type)
all_model=models.getmodel(47,'all',model_type)


ans={}

# Train and test

def train(num_epoch, model,loader,model_name,test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    last_loss=5.0
    for epoch in range(0, num_epoch):
        current_loss = 0.0
        current_corrects = 0
        losses = []
        model.train()
        loop = tqdm(enumerate(loader), total=len(loader)) # create a progress bar
        for batch_idx, (data, targets) in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            losses.append(loss)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(scores, 1)
            current_loss += loss.item() * data.size(0)
            current_corrects += (preds == targets).sum().item()
            accuracy = int(current_corrects / len(loader.dataset) * 100)
            loop.set_description(f"Epoch {epoch+1}/{num_epoch}")
            loop.set_postfix(loss=loss.data.item())
            # if batch_idx%50==0:
            #     test(model,test_loader)
        
        # save model
        torch.save(model.state_dict(), model_name+'.pt')


        
# model.eval() is a kind of switch for some specific layers/parts of the model that behave differently,
# during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. 
# You need to turn off them during model evaluation, and .eval() will do it for you. In addition, 
# the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() 
# to turn off gradients computation:
        
def test(model,loader,model_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, predictions = torch.max(output, 1)
            correct += (predictions == y).sum().item()
            
    test_loss /= len(loader.dataset)
    print("Average Loss: ", test_loss, "  Accuracy: ", correct, " / ",
    len(loader.dataset), "  ", round(correct / len(loader.dataset) * 100,2), "%")
    ans[model_name]=round(correct / len(loader.dataset) * 100,2)

def test_all(loader):
    model.eval()
    cat_model.eval()
    dog_model.eval()
    bird_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y,detail_label in tqdm(loader,total=len(loader)):
            x.to(device)
            y.to(device)
            detail_label.to(device)
            output = model(x)
            cat_output=cat_model(x)
            dog_output=dog_model(x)
            bird_output=bird_model(x)
            _, predictions = torch.max(output, 1)
            _, cat_predictions = torch.max(cat_output, 1)
            _, dog_predictions = torch.max(dog_output, 1)
            _, bird_predictions = torch.max(bird_output, 1)
            for i in range(len(predictions)):
                if(predictions[i] == 0):
                    if(cat_predictions[i] == detail_label[i]):
                        correct += 1
                elif (predictions[i]==1):
                    if(dog_predictions[i]+12 == detail_label[i]):
                        correct += 1
                elif (predictions[i]==2):
                    if(bird_predictions[i]+37 == detail_label[i]):
                        correct += 1
            
    print("Average Loss: ", test_loss, "  Accuracy: ", correct, " / ",
    len(loader.dataset), "  ", round(correct / len(loader.dataset) * 100,2), "%")
    ans['ours']=round(correct / len(loader.dataset) * 100,2)

def train_cat(epoch):
    cat_info=open('annotations/list.txt')
    cat_imgs=[]
    cat_labels=[]
    for row in cat_info:
        l=row.split(' ')
        if l[2]=='1':
            cat_imgs.append(f'annotations/images/{l[0]}.jpg')
            cat_labels.append(int(l[3])-1)
    cat_train_data, cat_test_data, cat_train_label, cat_test_label = train_test_split(cat_imgs, cat_labels, test_size=0.2, random_state=42)
    print("loading cat:")
    cat_train_dataset= [(cat_train_data[i], cat_train_label[i]) for i in range(len(cat_train_data))]
    cat_test_dataset= [(cat_test_data[i], cat_test_label[i]) for i in range(len(cat_test_data))]
    cat_train_dataset = models.ImageLoaderRAM(cat_train_dataset, train_transform,train=True)
    cat_test_dataset = models.ImageLoaderRAM(cat_test_dataset, test_transform,train=False)
    cat_train_loader = models.DataLoader(cat_train_dataset, batch_size=bs, shuffle=True)
    cat_test_loader = models.DataLoader(cat_test_dataset, batch_size=bs, shuffle=False)
    train(epoch,cat_model,cat_train_loader,'cat',cat_test_loader)
    test(cat_model,cat_test_loader,'cat')

def train_bird(epoch):
    imgs=[]
    labels=[]
    for i in range(10):
        kind=os.listdir('CUB_200_2011/CUB_200_2011/images')[i]
        for img in os.listdir(f'CUB_200_2011/CUB_200_2011/images/{kind}'):
            imgs.append(f'CUB_200_2011/CUB_200_2011/images/{kind}/{img}')
            labels.append(i)
    train_data, test_data, train_label, test_label = train_test_split(imgs, labels, test_size=0.2, random_state=42)
    print("loading bird")
    train_dataset= [(train_data[i], train_label[i]) for i in range(len(train_data))]
    test_dataset= [(test_data[i], test_label[i]) for i in range(len(test_data))]
    train_dataset = models.ImageLoaderRAM(train_dataset, train_transform,train=True)
    test_dataset = models.ImageLoaderRAM(test_dataset, test_transform,train=False)
    train_loader = models.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = models.DataLoader(test_dataset, batch_size=bs, shuffle=False)
    train(epoch, bird_model,train_loader,'bird',test_loader) #[64, 3, 224, 224])
    test(bird_model,test_loader,'bird')


def train_dog(epoch):
    dog_info=open('annotations/list.txt')
    dog_imgs=[]
    dog_labels=[]
    for row in dog_info:
        l=row.split(' ')
        if l[2]=='2':
            dog_imgs.append(f'annotations/images/{l[0]}.jpg')
            dog_labels.append(int(l[3])-1)
    dog_train_data, dog_test_data, dog_train_label, dog_test_label = train_test_split(dog_imgs, dog_labels, test_size=0.2, random_state=42)
    print("loading dog")
    dog_train_dataset= [(dog_train_data[i], dog_train_label[i]) for i in range(len(dog_train_data))]
    dog_test_dataset= [(dog_test_data[i], dog_test_label[i]) for i in range(len(dog_test_data))]
    dog_train_dataset = models.ImageLoaderRAM(dog_train_dataset, train_transform,train=True)
    dog_test_dataset = models.ImageLoaderRAM(dog_test_dataset, test_transform,train=False)
    dog_train_loader = models.DataLoader(dog_train_dataset, batch_size=bs, shuffle=True)
    dog_test_loader = models.DataLoader(dog_test_dataset, batch_size=bs, shuffle=False)
    train(epoch, dog_model,dog_train_loader,'dog',dog_test_loader) #[64, 3, 224, 224])
    test(dog_model,dog_test_loader,'dog')

def train_all(epoch):
    info=open('annotations/list.txt')
    imgs=[]
    labels=[]
    for row in info:
        l=row.split(' ')
        imgs.append(f'annotations/images/{l[0]}.jpg')
        labels.append(int(l[3])-1)
    for i in range(10):
        kind=os.listdir('CUB_200_2011/CUB_200_2011/images')[i]
        for img in os.listdir(f'CUB_200_2011/CUB_200_2011/images/{kind}'):
            imgs.append(f'CUB_200_2011/CUB_200_2011/images/{kind}/{img}')
            labels.append(i+37)
    train_data, test_data, train_label, test_label = train_test_split(imgs, labels, test_size=0.2, random_state=42)
    print("loading all")
    train_dataset= [(train_data[i], train_label[i]) for i in range(len(train_data))]
    test_dataset= [(test_data[i], test_label[i]) for i in range(len(test_data))]
    train_dataset = models.ImageLoaderRAM(train_dataset, train_transform,train=True)
    test_dataset = models.ImageLoaderRAM(test_dataset, test_transform,train=False)
    train_loader = models.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = models.DataLoader(test_dataset, batch_size=bs, shuffle=False)
    train(epoch, all_model,train_loader,'all',test_loader) #[64, 3, 224, 224])
    test(all_model,test_loader,'all')

def train_first(epoch):
    info=open('annotations/list.txt')
    imgs=[]
    labels=[]
    for row in info:
        l=row.split(' ')
        imgs.append(f'annotations/images/{l[0]}.jpg')
        labels.append(int(l[2])-1)
    for i in range(10):
        kind=os.listdir('CUB_200_2011/CUB_200_2011/images')[i]
        for img in os.listdir(f'CUB_200_2011/CUB_200_2011/images/{kind}'):
            imgs.append(f'CUB_200_2011/CUB_200_2011/images/{kind}/{img}')
            labels.append(2)
    train_data, test_data, train_label, test_label = train_test_split(imgs, labels, test_size=0.2, random_state=42)
    train_dataset=[(train_data[i],train_label[i]) for i in range(len(train_data))]
    test_dataset=[(test_data[i],test_label[i]) for i in range(len(test_data))]
    print("loading general data:")
    train_dataset = models.ImageLoaderRAM(train_dataset, train_transform,train=True)
    test_dataset = models.ImageLoaderRAM(test_dataset, test_transform,train=False)

    train_loader = models.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = models.DataLoader(test_dataset, batch_size=bs, shuffle=True)
    
    train(epoch, models,train_loader,'first',test_loader) # train
    test(models,test_loader,'first') # test

def test_all_init():
    cat_info=open('annotations/list.txt')
    cat_imgs=[]
    cat_labels=[]
    for row in cat_info:
        l=row.split(' ')
        if l[2]=='1':
            cat_imgs.append(f'annotations/images/{l[0]}.jpg')
            cat_labels.append(int(l[3])-1)
    cat_train_data, cat_test_data, cat_train_label, cat_test_label = train_test_split(cat_imgs, cat_labels, test_size=0.2, random_state=42)
    dog_info=open('annotations/list.txt')
    dog_imgs=[]
    dog_labels=[]
    for row in dog_info:
        l=row.split(' ')
        if l[2]=='2':
            dog_imgs.append(f'annotations/images/{l[0]}.jpg')
            dog_labels.append(int(l[3])-1)
    dog_train_data, dog_test_data, dog_train_label, dog_test_label = train_test_split(dog_imgs, dog_labels, test_size=0.2, random_state=42)
    imgs=[]
    labels=[]
    for i in range(10):
        kind=os.listdir('CUB_200_2011/CUB_200_2011/images')[i]
        for img in os.listdir(f'CUB_200_2011/CUB_200_2011/images/{kind}'):
            imgs.append(f'CUB_200_2011/CUB_200_2011/images/{kind}/{img}')
            labels.append(i)
    train_data, test_data, train_label, test_label = train_test_split(imgs, labels, test_size=0.2, random_state=42)
    print("load alltester:")
    all_test_data=cat_test_data+dog_test_data
    all_test_label=[0]*len(cat_test_data)+[1]*len(dog_test_data)+[2]*len(train_data)
    all_test_label_detail=cat_test_label+[dog_test_label[i]+12 for i in range(len(dog_test_label))]+[test_label[i]+37 for i in range(len(test_label))]
    all_test_dataset=[(all_test_data[i], all_test_label[i]) for i in range(len(all_test_data))]
    all_test_dataset = models.AllImageLoader(all_test_dataset,all_test_label_detail, test_transform)
    all_test_loader = models.DataLoader(all_test_dataset, batch_size=64, shuffle=False)
    test_all(all_test_loader)
