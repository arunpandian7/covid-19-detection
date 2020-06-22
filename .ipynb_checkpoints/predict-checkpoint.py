import torch
import torch.nn as nn
from torchvision import models

from PIL import Image
import numpy as np


PATH = './model/model.pt'
device = torch.device('cpu')
class_names = ['covid19', 'normal']

#Defining Model Architecture
def CNN_Model(pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))
    model.to(torch.device('cpu'))
    return model
model = CNN_Model(pretrained=False)

#Loading the Model Trained Weights
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval() # Setting to Evaluation Mode to get Inference


#reading an image
img = Image.open('./test/covid_test1.jpg')
img = img.resize((224,224))
img_np = np.array(img)
img_tensor = torch.tensor(img_np)

pred = model(img_tensor)

print(pred)

