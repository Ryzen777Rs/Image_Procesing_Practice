# training/src/train.py
import os
import torch
from torch.utils.data import DataLoader

# Importăm piesele construite de noi în fișierele anterioare
from model import get_model
from dataset import CustomDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"--- Se rulează antrenarea pe: {device} ---")
    if device.type == 'cuda':
        print(f"Placă detectată: {torch.cuda.get_device_name(0)}")


    NUM_CLASSES = 1      
    BATCH_SIZE = 4       
    EPOCHS = 10          
    LEARNING_RATE = 0.005

    

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    if not os.path.exists(os.path.join(data_dir, "images")):
        print(f"⚠️ EROARE: Nu am găsit folderul cu imagini la calea: {os.path.abspath(data_dir)}/images")
        print("Creează folderul și pune câteva poze înainte de a rula antrenarea.")
        return

    dataset = CustomDataset(data_dir)
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )


    model = get_model(num_classes=NUM_CLASSES)
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    
    model.train() 
    
    print("\n🚀 Începe antrenarea...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        
        for images, targets in data_loader:
        
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoca [{epoch+1}/{EPOCHS}] - Pierdere totală (Loss): {epoch_loss:.4f}")

    print("\n✅ Antrenare finalizată cu succes!")


    
    print("📦 Se exportă modelul în format model.onnx pentru aplicația din Rust...")
    model.eval() 
    dummy_input = [torch.randn(3, 800, 800).to(device)] 
    
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "model.onnx")
    
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            opset_version=16, 
            input_names=["images"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes={
                "images": {0: "batch_size", 2: "height", 3: "width"} 
            }
        )
    print(f"🎉 Succes! Fișierul salvat se află la: {os.path.abspath(onnx_path)}")

if __name__ == "__main__":
    main()