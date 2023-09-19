from ultralytics import YOLO
import matplotlib.pyplot as plt

MODEL_PATH = '/home/faraujo/TFM/yolo_trainings/train_100_include02_large/best.pt'

model = YOLO(MODEL_PATH)

image_path = '/home/faraujo/TFM/datasets/datasets_agosto/validation/images/LIDC-IDRI-0002_189.png'

# results = model(image_path, save = False)

predictions = model.predict(image_path, conf=0.01)
img_plot = predictions[0].plot()


print(dir(predictions[0].masks))
plt.imshow(img_plot)
plt.show()





