import json
import csv

with open('example_gt.json', 'r') as f:
    data = json.load(f)

header = ['image_id']
parts = ['body', 'lefthand', 'righthand', 'face', 'foot']
for part in parts:
    for i in range(0, 133):  # up to 133 possible keypoints (some are shorter)
        header.extend([f'{part}_x_{i}', f'{part}_y_{i}', f'{part}_v_{i}'])

with open('example_pred.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for entry in data:
        row = [entry['image_id']]
        for part in parts:
            kpts = entry.get(f'{part}_kpts', [])
            for i in range(0, len(kpts), 3):
                row.extend(kpts[i:i+3])
            # Pad missing keypoints
            missing = 133 - len(kpts)//3
            row.extend([0, 0, 0] * missing)
        writer.writerow(row)

print("✅ CSV file created: example_pred.csv")
