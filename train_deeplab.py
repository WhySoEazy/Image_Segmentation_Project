from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim import SGD
from torchvision.datasets import VOCSegmentation
from VOC_seg_dataset import MyVOCSeg
import tqdm
from tqdm import tqdm
from argparse import ArgumentParser
import argparse
import os
import shutil
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy , MulticlassJaccardIndex
from torchmetrics.segmentation import MeanIoU

def get_args():
    parse = ArgumentParser(description="DeepLab Training")

    parse.add_argument("--data_path", 
                       type=str, 
                       default="data")
    
    parse.add_argument("--epochs",
                       type=int,
                       default=100)
    
    parse.add_argument("--batchs",
                       type=int,
                       default=8)
    
    parse.add_argument("--lr",
                       type=float,
                       default=1e-3)
    
    parse.add_argument("--momentum",
                       type=float,
                       default=0.9)
    
    parse.add_argument("--logging",
                       type=str,
                       default="tensorboard")
    
    parse.add_argument("--checkpoint",
                       type=str,
                       default=None)
    
    parse.add_argument("--trained_model",
                       type=str,
                       default="trained_model")

    parse.add_argument("--image_size",
                       type=int,
                       default=224)
    
    args = parse.parse_args()

    return args

def train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    num_epochs = args.epochs

    transform = Compose([
        Resize((args.image_size,args.image_size)),
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
    ])

    target_transform = Resize((args.image_size,args.image_size))

    train_dataset = MyVOCSeg(root=args.data_path,
                             year="2012",
                             image_set="train",
                             download=False,
                             transform=transform,
                             target_transform=target_transform)
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchs,
        num_workers=4,
        shuffle=True,
        drop_last=False
    )

    test_dataset = MyVOCSeg(root=args.data_path,
                             year="2012",
                             image_set="val",
                             download=False,
                             transform=transform,
                             target_transform=target_transform)
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batchs,
        num_workers=4,
        drop_last=False
    )

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True).to(device)
    optimizer = SGD(lr=args.lr , momentum=args.momentum , params=model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_iou = checkpoint["best_map"]
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        best_iou = 0

    if not os.path.isdir(args.trained_model):
            os.mkdir(args.trained_model)

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging , ignore_errors=True)

    writer = SummaryWriter(args.logging)

    num_iters = len(train_dataloader)
    accuracy_metric = MulticlassAccuracy(num_classes=len(train_dataset.classes)).to(device)
    mean_iou_metric = MulticlassJaccardIndex(num_classes=len(train_dataset.classes)).to(device)
    for epoch in range(start_epoch , start_epoch + args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader)
        all_losses = []
        for iter , (images , targets) in enumerate(progress_bar):
            #Forward
            images = images.to(device)
            targets = targets.to(device)
            result = model(images)
            output = result["out"]
            loss = criterion(output , targets)
            all_losses.append(loss.item())
            avg_loss = np.mean(all_losses)
            progress_bar.set_description("Epoch: {}/{} , Loss: {:0.4f}".format(epoch+1 , args.epochs , avg_loss))
            #Tensorboard
            writer.add_scalar("Train/Loss" , avg_loss , epoch * num_iters + iter)
            #Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #evaluation
        model.eval()
        progress_bar = tqdm(test_dataloader)
        test_acc = []
        test_miou = []

        with torch.no_grad():
            for images , targets in progress_bar:
                images = images.to(device)
                targets = targets.to(device)
                result = model(images)
                output = result["out"]
                accuracy = accuracy_metric(output , targets).item()
                miou = mean_iou_metric(output , targets).item()
                test_acc.append(accuracy)
                test_miou.append(miou)

        avg_acc = np.mean(test_acc)
        avg_miou = np.mean(test_miou)
        print("Accuracy: {} , mIoU: {}".format(avg_acc , avg_miou))
        writer.add_scalar("Valid/Accuracy" , avg_acc , epoch)
        writer.add_scalar("Valid/mIoU" , avg_miou , epoch)

        #SaveModel
        if miou > best_iou:
            best_iou = miou
            checkpoint = {
                "best_iou": best_iou,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_deeplab.pt".format(args.trained_model))

        checkpoint = {
            "epoch": epoch + 1,
            "best_map": best_iou,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_deeplab.pt".format(args.trained_model))

        print("Best mIoU: {}".format(best_iou))        


            
if __name__ == "__main__":
    args = get_args()
    train(args)