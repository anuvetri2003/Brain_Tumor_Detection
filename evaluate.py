from ultralytics import YOLO # type: ignore

model = YOLO("best.pt")

metrics = model.val(data="data.yaml")  # un dataset yaml path

print(metrics)