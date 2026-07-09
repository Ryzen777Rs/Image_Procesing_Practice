import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDatatset(Dataset):     #mosteneste o structuradin torch(traduce pozele pentru placa video)
    def __init__(self, data_dir):  #constructor
        self.data_dir =data_dir
        self.image_files =sorted([f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(('.jpg','.jpeg','.png'))])  #sortare poze

    def __len__(self):  
        return len(self.image_files)  #nr de poze
    
    def __getitem__(self,idx):       #gasim imaginea in calculator
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, "images", img_name)
        img = Image.open(img_path).convert("RGB") 

        from torchvision.transforms import functional as Fun
        img_tensor = Fun.to_tensor(img)

        
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.data_dir, "labels", label_name)

        boxes = []
        labels = []

        # citirea fisierului text pentru coordonate box
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0]) 
                        
                        # Coordonatele absolute 
                        xmin = float(parts[1])
                        ymin = float(parts[2])
                        xmax = float(parts[3])
                        ymax = float(parts[4])
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        
                        labels.append(class_id + 1) 

        # daca nu exista obiecte
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        return img_tensor, target
