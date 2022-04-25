from jetcam.usb_camera import USBCamera
import ipywidgets
import traitlets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg
import threading
import time
from utils import preprocess
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataset import ImageClassificationDataset
import os
import IPython

datasets = {}
for name in DATASETS:
    datasets[name] = ImageClassificationDataset('../data/classification/' + TASK + '_' + name, CATEGORIES, TRANSFORMS)

CATEGORIES = ['Familiar', 'Stranger']
DATASETS = ['F', 'S']

camera = USBCamera(width=224, height=224, capture_device=0)
image = camera.value
camera.running = True
project = True

device = torch.device('cuda')

# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(dataset.categories))
    
model = model.to(device)
camera_widget = ipwidgets.Immage()
#model_load_button = ipywidgets.Button(description='load model')
#model_path_widget = ipywidgets.Text(description='Model/Path', value='/nvdli-nano/data/classification/face_model.pth')

#def load_model(c):
    #model.load_state_dict(torch.load(model_path_widget.value))
#model_load_button.on_click(load_model)

#model_widget = ipywidgets.VBox([
    #model_path_widget,
    #ipywidgets.HBox([model_load_button])
#])

prediction_widget = ipywidgets.Text(description='Detected')
score_widgets = []

def live(model, camera, prediction_widget):
    global dataset
    while project:
        image = camera.value
        preprocessed = preprocess(image)
        output = model(preprocessed)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        prediction_widget.value = dataset.categories[category_index]
        
live_execution_widget = ipywidgets.VBox([
    ipywidgets.HBox(score_widgets),
    prediction_widget
])

all_widgets = ipywidgets.VBox([
    ipywidgets.HBox([live_execution_widget]),
    camera_widget
])

display(all_widgets)

time.sleep(100)
os._exit(00)

