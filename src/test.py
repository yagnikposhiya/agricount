"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from config.config import Config
from torchvision import transforms

if __name__ == '__main__':

    config = Config() # create an instance of config class

    # load a pre-trained SSD model
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()

    image = Image.open(config.SAMPLE_IMAGE_PATH) # load an image
    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor()
    ]) # transform a loaded image

    image = transform(image).unsqueeze(0) # add batch dimension

    # perform inference
    with torch.no_grad():
        predictions = model(image)

    # extract predictions
    print(predictions)

    # Extract predictions
    predicted_boxes = predictions[0]['boxes'].cpu().numpy()
    predicted_scores = predictions[0]['scores'].cpu().numpy()
    predicted_labels = predictions[0]['labels'].cpu().numpy()

    # process input image
    image = image.squeeze(0) # remove batch dimension
    print("- Shape of an image: {}".format(image.shape))
    print("- Type of an image: {}".format(type(image)))
    numpy_image = image.cpu().detach().numpy() # move tensor to CPU if it is on GPU, detach tensor if it requires gradients, and convert it to numpy array
    print("- Shape of a numpy image: {}".format(numpy_image.shape))
    print("- Type of a numpy image: {}".format(type(numpy_image)))
    numpy_image = np.transpose(numpy_image,(1,2,0)) # change shape of ndarray from (3,300,300) to (300,300,3)
    

    # Visualize the results
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20)) # create figure canvas 
    ax.imshow(numpy_image)

    # Define a threshold to filter out low-confidence detections
    confidence_threshold = 0.1

    for box, score, label in zip(predicted_boxes, predicted_scores, predicted_labels):
        if score > confidence_threshold:
            # Draw bounding box
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Draw label and score
            ax.text(x_min, y_min - 10, f'{label}: {score:.2f}', color='red', fontsize=12, backgroundcolor='none')

    plt.show()