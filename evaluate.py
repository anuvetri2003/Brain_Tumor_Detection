from ultralytics import YOLO # type: ignore
model = YOLO("best.pt")
metrics = model.val(data="data.yaml")
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
map50 = metrics.results_dict['metrics/mAP50(B)']
map95 = metrics.results_dict['metrics/mAP50-95(B)']
f1 = 2 * (precision * recall) / (precision + recall)
print("\n----- CLEAN METRICS -----")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"mAP@50     : {map50:.4f}")
print(f"mAP@50-95  : {map95:.4f}")
print("-------------------------\n")
print("Confusion Matrix saved in: runs/detect/")