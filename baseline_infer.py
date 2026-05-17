from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--test-dir", default="test/images", help="Directory of test images")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    args = parser.parse_args()

    model = YOLO(args.model)
    image_dir = Path(args.test_dir)
    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file()])

    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()

        for img_path in image_paths:
            results = model.predict(
                source=str(img_path), 
                conf=args.conf, 
                save=False, 
                verbose=False,
                augment=args.tta,  # Enable TTA
                fliplr=0.5 if args.tta else 0.0  # Horizontal flip for TTA
            )
            if not results or results[0].boxes is None:
                continue
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    writer.writerow(
                        {
                            "image_id": img_path.name,
                            "class_id": int(box.cls[0].item()),
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "confidence": float(box.conf[0].item()),
                        }
                    )


if __name__ == "__main__":
    main()