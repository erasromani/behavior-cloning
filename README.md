# Car Racing
## 1.0 Introduction
For this exercise, we collected data from the openai gym “CarRacing-v0” gaming interface. This was done by playing the game various times. The data includes the following:
* State – 96 x 96 pixel RGB image of each frame
* Action – three element vector denoting the set of actions taken corresponding to the frame 

Below is an example of the data collected. Note that the action associate with the frame is [-1, 0, 0] which denotes LEFT.

![alt text](https://github.com/erasromani/car-racing/blob/main/images/example_data.png)

## 1.1 Data Processing

Given that actions are velocity dependent, we would like to restructure the for of the data such that the network has some signal of the velocity. This was done by first turning the images to grayscale, then stacking the past 5 frames to form a 5 x 96 x 96 rank 3 tensor. Below is an example of the input tensor. 

![alt text](https://github.com/erasromani/car-racing/blob/main/images/input_tensor.png)

We also made some modifications to the form of the target variable, the action. It is given by a two-element vector [steer, accelerate] such that
* steer = {“NOTHING”: 0, “LEFT”: 1, “RIGHT”, 2}
* accelerate = {“NOTHING”: 0, “ACCELERATE”: 1, “BRAKE”: 2}

## 1.2 Network Architecture

Our neural network starts with a CNN network to extract visual / temporal features, followed by two subnetworks each consisting of a two layer fully connected network. The first and second subnetwork is dedicated to predicting the probability of each component steer and accelerate respectively.

![alt text](https://github.com/erasromani/car-racing/blob/main/images/network_architecture.png)

## 1.3 Loss Function

The loss function is given by the weighted addition of two cross-entropy loss evaluated following each subnetwork and is expressed as follows

![alt text](https://latex.codecogs.com/gif.latex?L%20%3D%20L_%7Bsteer%7D%20&plus;%20%5Clambda%20L_%7Baccelerate%7D)


where ![alt text](https://latex.codecogs.com/gif.latex?L), ![alt text](https://latex.codecogs.com/gif.latex?L_%7Bsteer%7D), and ![alt text](https://latex.codecogs.com/gif.latex?L_%7Baccelerate%7D)  is the total loss, cross-entropy loss for the steer subnetwork, and the cross-entropy loss for the accelerate subnetwork. 𝜆 is a scaling factor which is set empirically to ensure each component of the loss are of similar scale. Based on our observations, 𝜆=1.0 yields comparable scales between the steering and acceleration loss components.
