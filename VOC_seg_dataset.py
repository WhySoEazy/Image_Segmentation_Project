from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np

class MyVOCSeg(VOCSegmentation):
    def __init__(self , root , year , image_set , download , transform=None):
        super().__init__(root, year, image_set, download)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                        'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse', 
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.transform = transform

    def __getitem__(self , item):
        image , target = super().__getitem__(item)
        target = np.array(target)
        target[target==255] = 0
        if self.transform != None:
            image = self.transform(image)
        return image , target

if __name__ == "__main__":
    transform = Compose([
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
    ])

    dataset = MyVOCSeg(root="data", 
                       year="2012", 
                       image_set="train",
                       download=False,
                       transform=transform)
    
    image , target = dataset[1000]
    print(image)
    print(np.unique(target))