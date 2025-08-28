from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim import SGD
from torchvision.datasets import VOCSegmentation
from VOC_seg_dataset import MyVOCSeg
import tqdm
from tqdm import tqdm

def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    num_epochs = 100

    transform = Compose([
        Resize((224,224)),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
    ])

    target_transform = Resize((224,224))

    train_dataset = MyVOCSeg(root="data",
                             year="2012",
                             image_set="train",
                             download=False,
                             transform=transform,
                             target_transform=target_transform)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        drop_last=False
    )

    test_dataset = MyVOCSeg(root="data",
                             year="2012",
                             image_set="val",
                             download=False,
                             transform=transform,
                             target_transform=target_transform)
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=4,
        drop_last=False
    )

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to(device)
    optimizer = SGD(lr=1e-3 , momentum=0.9 , params=model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        progess_bar = tqdm(train_dataloader)
        #Forward
        for images , targets in progess_bar:
            images = images.to(device)
            targets = targets.to(device)
            result = model(images)
            output = result["out"]
            loss = criterion(output , targets)
            progess_bar.set_description("Epoch: {}/{} , Loss: {:0.4f}".format(epoch+1 , num_epochs , loss.item()))
            #Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
if __name__ == "__main__":
    train()