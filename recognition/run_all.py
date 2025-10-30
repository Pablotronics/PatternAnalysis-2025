import argparse, subprocess, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to keras_slices_data (or its parent)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--save_pngs", action="store_true")
    ap.add_argument("--num_samples", type=int, default=9)
    args = ap.parse_args()

    here = os.path.dirname(__file__)

    # Train
    train_cmd = [
        sys.executable,
        os.path.join(here, "Project2", "train.py"),
        "--data_root", args.data_root,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
    ]
    print("[INFO] Running:", " ".join(train_cmd))
    subprocess.check_call(train_cmd)

    # Predict
    predict_cmd = [
        sys.executable,
        os.path.join(here, "Project2", "predict.py"),
        "--data_root", args.data_root,
        "--num_samples", str(args.num_samples),
    ]
    if args.save_pngs: predict_cmd.append("--save_pngs")
    print("[INFO] Running:", " ".join(predict_cmd))
    subprocess.check_call(predict_cmd)

if __name__ == "__main__":
    main()
