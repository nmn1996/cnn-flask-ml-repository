This setup will serve a simple CNN model via an HTTP endpoint (/predict). You can send an image file to this endpoint as a POST request, and it will return the predicted digit.

Below are the steps for the overall deployment and accessing the model intereference locally :

1. We have two scripts cnn-mode.py and interference.py .

2. The cnn-model.py python script trains a simple Convolutional Neural Network (CNN) model on the MNIST dataset, which consists of handwritten digit images. After training, it saves the trained model's state dictionary to a file named mnist_model.pt. This file contains the learned parameters of the model, which can be later loaded for inference or further training.

3. The interference.py python script defines a Flask web application that serves as a simple API for making predictions on handwritten digit images using a Convolutional Neural Network (CNN) model trained on the MNIST dataset. This script creates a web server that listens for incoming HTTP requests containing images, processes these images using a pre-trained CNN model, and returns the predicted digits as JSON responses.

4. We then create docker image from the Dockerfile as per below .

![Alt text](/Users/guptanam/Desktop/ml-images/Dockerbuild.png)

5.
