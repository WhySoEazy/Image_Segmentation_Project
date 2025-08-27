from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
from VOC_seg_dataset import MyVOCSeg
from torch.utils.data import DataLoader
import torch
from torch.optim import SGD

def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    num_epochs = 100

    transform = Compose([
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
    ])

    train_dataset = MyVOCSeg(root="data",
                             year="2012",
                             image_set="train",
                             download=False,
                             transform=transform)
    
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
                             download=False,)
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=4,
        drop_last=False
    )

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    optimizer = SGD(lr=1e-3 , momentum=0.9 , params=model.parameters())

    for epoch in range(num_epochs):
        model.train()
        

if __name__ == "__main__":
    train()