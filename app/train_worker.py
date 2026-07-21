import argparse
import os
import sys
import time


def check_device():
    """Detectează dacă o placă video NVIDIA este disponibilă pentru antrenare."""
    try:
        import torch

        if torch.cuda.is_available():
            print(
                "STATUS:INFO|Placă NVIDIA detectată. Antrenarea se va face pe GPU!"
            )
            return "0"  # ID-ul plăcii video CUDA
    except ImportError:
        pass
    print(
        "STATUS:INFO|Nu s-a detectat placă NVIDIA. Antrenarea se va face pe procesor (CPU)."
    )
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Worker separat pentru antrenarea YOLO"
    )

    # Argumente necesare pentru antrenare
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

    # Callback apelat automat de Ultralytics YOLO la finalul fiecărei epoci
    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        total_epoci = trainer.epochs
        procentaj = int((epoch / total_epoci) * 100)

        # Calcul timp estimat rămas
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

        # Transmitere status către aplicația GUI
        print(
            f"PROGRESS:{procentaj}|PERCENT:{procentaj}%|REMAIN:{timp_ramas_str}"
        )
        sys.stdout.flush()

    try:
        from ultralytics import YOLO

        # Determinare model de pornire
        cale_model = (
            "yolov8n.pt"
            if args.model == "Model YOLO neantrenat" or not args.model
            else args.model
        )
        nume_proiect = args.new_name if args.new_name else "Model_Nou"

        # Verificare dacă fișierul de date există
        if not os.path.exists(args.data):
            print(
                f"STATUS:ERROR|Fișierul de date '{args.data}' nu a fost găsit!"
            )
            sys.stdout.flush()
            sys.exit(1)

        print(f"STATUS:INFO|Încărcare model: {cale_model}...")
        sys.stdout.flush()

        # Inițializare model
        model = YOLO(cale_model)

        # Atașare callback pentru bara de progres
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        print(
            f"STATUS:INFO|Pornire antrenare: {cale_model} -> {nume_proiect} ({args.epochs} epoci)"
        )
        sys.stdout.flush()

        # ANTRENAREA REALĂ YOLO
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            lr0=args.lr,
            workers=args.workers,
            device=device,
            project="models",  # Folderul unde se vor salva rezultatele
            name=nume_proiect,  # Numele subfolderului
            exist_ok=True,  # Permite rescrierea/continuarea în același folder
        )

        print("STATUS:DONE")
        sys.stdout.flush()

    except ImportError:
        print(
            "STATUS:ERROR|Biblioteca 'ultralytics' nu este instalată! Rulează: pip install ultralytics"
        )
        sys.stdout.flush()
        sys.exit(1)
    except Exception as e:
        print(f"STATUS:ERROR|A apărut o eroare în timpul antrenării: {str(e)}")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()