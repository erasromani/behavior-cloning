# Car Racing
## 1.0 Introduction
For this exercise, we collected data from the openai gym “CarRacing-v0” gaming interface. This was done by playing the game various times. The data includes the following:
* State – 96 x 96 pixel RGB image of each frame
* Action – three element vector denoting the set of actions taken corresponding to the frame 

Below is an example of the data collected. Note that the action associate with the frame is [-1, 0, 0] which denotes LEFT.

![example data](https://github.com/erasromani/car-racing/blob/main/images/example_data.png)

## 1.1 Data Processing

Given that actions are velocity dependent, we would like to restructure the for of the data such that the network has some signal of the velocity. This was done by first turning the images to grayscale, then stacking the past 5 frames to form a 5 x 96 x 96 rank 3 tensor. Below is an example of the input tensor. 

![input tensor](https://github.com/erasromani/car-racing/blob/main/images/input_tensor.png)

We also made some modifications to the form of the target variable, the action. It is given by a two-element vector [steer, accelerate] such that
* steer = {“NOTHING”: 0, “LEFT”: 1, “RIGHT”, 2}
* accelerate = {“NOTHING”: 0, “ACCELERATE”: 1, “BRAKE”: 2}

## 1.2 Network Architecture

Our neural network starts with a CNN network to extract visual / temporal features, followed by two subnetworks each consisting of a two layer fully connected network. The first and second subnetwork is dedicated to predicting the probability of each component steer and accelerate respectively.

![network architecture](https://github.com/erasromani/car-racing/blob/main/images/network_architecture.png)

## 1.3 Loss Function

The loss function is given by the weighted addition of two cross-entropy loss evaluated following each subnetwork and is expressed as follows

L = L<sub>steer</sub> + 𝜆 L<sub>accelerate</sub>

where L, L<sub>steer</sub>, and L<sub>accelerate</sub> is the total loss, cross-entropy loss for the steer subnetwork, and the cross-entropy loss for the accelerate subnetwork. 𝜆 is a scaling factor which is set empirically to ensure each component of the loss are of similar scale. Based on our observations, 𝜆=1.0 yields comparable scales between the steering and acceleration loss components.

## 1.4 Training 

We evaluated the impact of training set size on performance by assessing the validation loss, accuracy, and agent 10 episode average score at different training set size values. Note that as the training set size increases, the accuracy increases while the loss decreases as expected. The agent score over training set size also show a similar trend.  

![sample size evaluation](https://github.com/erasromani/car-racing/blob/main/images/sample_size_evaluation.PNG)

## 1.5 Lessons Learned
<ol type="a">
  <li>In training the agent, we utilized the learning rate finder method described in [L. Smith. Cyclic Learning Rates for Training Neural Networks. arXiv preprint arXiv:1506.01186, 2015](https://arxiv.org/pdf/1506.01186.pdf) to find an appropriate learning rate. Below is an example figure of the learning rate finder process. The point at which the training loss decreases at the highest rate yields an appropriate learning rate./li>
  <li>Tea</li>
  <li>Milk</li>
</ol>

