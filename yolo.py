from ultralytics import YOLO


model = YOLO('./veg.pt')
results = model.track(source=0, show=True, tracker='bytetrack.yaml')