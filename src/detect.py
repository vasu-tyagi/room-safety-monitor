# detect.py
# Runs person detection on a folder of .png frames, draws boxes + a people-count,
# and writes a per-frame log. This is the cheapest layer of the system:
# small, fast person detection that could run continuously at the edge.
#
# Usage:
#   python detect.py <folder_of_frames> <output_folder>

import sys, os, csv
import cv2
from ultralytics import YOLO

input_dir = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

# yolov8n = "nano": the smallest, fastest YOLO model. Good enough to find people,
# cheap enough that running it on every frame is plausible at the edge.
# The weights (~6 MB) download automatically the first time.
model = YOLO("yolov8n.pt")

PERSON = 0  # in the COCO label set, class 0 is "person". We ignore everything else.

frames = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png"))
rows = []

for name in frames:
    img = cv2.imread(os.path.join(input_dir, name))

    # detect only people, only above 0.4 confidence
    result = model(img, classes=[PERSON], conf=0.4, verbose=False)[0]

    count = 0
    biggest = None  # we track the largest person box for the fall signal
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        w, h = x2 - x1, y2 - y1
        count += 1
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"person {conf:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if biggest is None or (w * h) > biggest[0]:
            biggest = (w * h, w, h)

    # stamp the occupancy count on the frame
    cv2.putText(img, f"people: {count}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, name), img)

    # log the cheap fall signal: aspect ratio (height / width) of the biggest person.
    # standing person -> tall box (ratio > 1). lying on the floor -> wide box (ratio < 1).
    # a fall shows up as this ratio dropping across consecutive frames.
    if biggest:
        _, w, h = biggest
        ratio = round(h / w, 2) if w else 0
        rows.append([name, count, round(w), round(h), ratio])
    else:
        rows.append([name, 0, "", "", ""])

with open(os.path.join(output_dir, "log.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "people", "box_width", "box_height", "aspect_ratio_h_over_w"])
    writer.writerows(rows)

print(f"Done. Annotated frames and log.csv are in: {output_dir}")
print(f"Processed {len(frames)} frames.")
