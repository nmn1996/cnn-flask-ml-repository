This setup will serve a simple CNN model via an HTTP endpoint (/predict). You can send an image file to this endpoint as a POST request, and it will return the predicted digit.

Below are the steps for the overall deployment and accessing the model intereference locally :

1. We have two scripts cnn-mode.py and interference.py .

2. The cnn-model.py python script trains a simple Convolutional Neural Network (CNN) model on the MNIST dataset, which consists of handwritten digit images. After training, it saves the trained model's state dictionary to a file named mnist_model.pt. This file contains the learned parameters of the model, which can be later loaded for inference or further training.

3. The interference.py python script defines a Flask web application that serves as a simple API for making predictions on handwritten digit images using a Convolutional Neural Network (CNN) model trained on the MNIST dataset. This script creates a web server that listens for incoming HTTP requests containing images, processes these images using a pre-trained CNN model, and returns the predicted digits as JSON responses.

4. We then create docker image from the Dockerfile .

5. Inside the docker image we run model-run.sh shell script to run both the above python script after the .pt file is generated from the cnn-model.py the flask app is run from interfference.py as per the logic defined in the model-run.sh.

6. Then the kubernetes deployment is run from deployment.yaml.

7. After that the node port service is run using service.yaml.

8. Then to access the service locally we do port forwarding of the service using kubectl port-forward service/my-model-service 12345:5000 to access the service locally on 12345 port.

9. Then fire POST curl request to test it where we provide the images in the POST request on which model is run and json response is provided. Below is a sample curl which puts the image as input from my local machine.

curl -X POST http://localhost:12345/predict -F "image=@/Users/guptanam/Downloads/gabriel-crismariu-IgQo1_gUzBY-unsplash.jpg"
