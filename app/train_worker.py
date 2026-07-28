import argparse
import os
import sys
import time


def check_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "0" 
    except ImportError:
        pass
    return "cpu"


def main():
    # 1. Redirecționarea output-ului către fișierul din folderul părinte
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    log_path = os.path.join(parent_dir, "antrenare_log.txt")

    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

    parser = argparse.ArgumentParser(description="Worker YOLO")
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Calea către fișierul dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Calea sau numele modelului"
    )
    parser.add_argument(
        "--new_name", type=str, default="Model_Nou", help="Numele proiectului de ieșire"
    )

    args = parser.parse_args()
    device = check_device()
    start_time = time.time()

    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        total_epoci = trainer.epochs
        procentaj = int((epoch / total_epoci) * 100)

        timp_scurs = time.time() - start_time
        if epoch > 1:
            timp_mediu_epoca = timp_scurs / epoch
            epoci_ramase = total_epoci - epoch
            secunde_ramase = int(epoci_ramase * timp_mediu_epoca)
            min_r = secunde_ramase // 60
            sec_r = secunde_ramase % 60
            timp_ramas_str = (
                f"remain {min_r}m {sec_r}s" if min_r > 0 else f"remain {sec_r}s"
            )
        else:
            timp_ramas_str = "calculare..."

        sys.stdout.flush()

    try:
        from ultralytics import YOLO
        import torch

        cale_model = (
            "yolov8n.pt"
            if args.model == "Model YOLO neantrenat" or not args.model
            else args.model
        )
        nume_proiect = args.new_name if args.new_name else "Model_Nou"
        if not os.path.exists(args.data):
            print(
                f"STATUS:ERROR|Fișierul de date '{args.data}' nu a fost găsit!"
            )
            sys.stdout.flush()
            sys.exit(1)

        print(f"STATUS:INFO|Încărcare model: {cale_model}...")
        sys.stdout.flush()
        model = YOLO(cale_model)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        sys.stdout.flush()

        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            lr0=args.lr,
            workers=args.workers,
            device=device,
            project="models",  
            name=nume_proiect,  
            exist_ok=True,  
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sys.stdout.flush()

    except ImportError as e:
        print(f"STATUS:ERROR|ImportError: {e}")
        sys.stdout.flush()
        sys.exit(1)
    except Exception as e:
        print(f"STATUS:ERROR|Exception: {e}")
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()