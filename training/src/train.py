# training/src/train.py
import os
import torch
from torch.utils.data import DataLoader

# Importăm piesele construite de noi în fișierele anterioare
from model import get_model
from dataset import CustomDatatset

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Se rulează antrenarea pe: {device} ---")
    if device.type == 'cuda':
        print(f"Placă detectată: {torch.cuda.get_device_name(0)}")

    NUM_CLASSES = 1      
    BATCH_SIZE = 4       
    EPOCHS = 150          # Poți mări numărul după ce vezi că exportul funcționează perfect
    LEARNING_RATE = 0.0005

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    if not os.path.exists(os.path.join(data_dir, "images")):
        print(f"⚠️ EROARE: Nu am găsit folderul cu imagini la calea: {os.path.abspath(data_dir)}/images")
        return

    dataset = CustomDatatset(data_dir)
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
            
            # Prevenim valorile de tip "nan" prin limitarea gradientului
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            epoch_loss += losses.item()

        print(f"Epoca [{epoch+1}/{EPOCHS}] - Pierdere totală (Loss): {epoch_loss:.4f}")

    print("\n✅ Antrenare finalizată cu succes!")

    print("📦 Se exportă modelul în format model.onnx...")
    model.eval() 

    # Pregătim un input dummy curat (listă de tensori, specific FasterRCNN)
    dummy_input = [torch.randn(3, 800, 800, dtype=torch.float32).to(device)]
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "model.onnx")

    # --- FORȚARE EXPORTATOR CLASIC (TORCHSCRIPT) ---
    # Pentru a opri noul exportator experimental (torch.export) care dă eroare pe FasterRCNN,
    # convertim mai întâi modelul într-un script traced.
    try:
        print("🔄 Se generează graful stabil prin TorchScript tracing...")
        traced_model = torch.jit.trace(model, (dummy_input,))
        
        with torch.no_grad():
            torch.onnx.export(
                traced_model,             # Pasăm modelul compilat prin JIT, nu modelul brut
                (dummy_input,),  
                onnx_path,
                opset_version=13,         # Opset 13/14 este extrem de stabil pentru TorchScript + FasterRCNN
                input_names=["images"],
                output_names=["boxes", "labels", "scores"],
                do_constant_folding=True
            )
        print(f"🎉 Succes absolut! Fișierul salvat se află la: {os.path.abspath(onnx_path)}")
        
    except Exception as e:
        print(f"⚠️ Tracing-ul direct a eșuat. Încercăm exportul clasic cu dezactivarea noului modul: {e}")
        # Plan de rezervă: dezactivăm complet rutele noi din opțiunile exportatorului
        from torch.onnx import _constants
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_path,
                opset_version=13,
                input_names=["images"],
                output_names=["boxes", "labels", "scores"],
                do_constant_folding=True,
                # Forțăm noul PyTorch să nu folosească rutele Dynamo/Export mapate în fundal
                dynamo=False if hasattr(torch.onnx, 'dynamo') else None 
            )
        print(f"🎉 Succes prin ruta de rezervă! Fișierul salvat se află la: {os.path.abspath(onnx_path)}")

if __name__ == "__main__":
    main()