# S8 Assignment

* As part of this assignment, models using Batch Normalization, Layer Normalization and Group Normalization were built and their results were provided below.
* As part of this repository, there are 5 files.
     * *model.py*  - models for session6, session7, batch normalization, layer normalization and group normalization.
     * *S8_BN.ipynb* - where we used Batch Normalization model ( Model_4) on CIFAR10 dataset and obtained desired results.
     * *S8_LN.ipynb* - where we used Layer Normalization model ( Model_5) on CIFAR10 dataset and obtained desired results.
     * *S8_GN.ipynb* - where we used Group Normalization model ( Model_6) on CIFAR10 dataset and obtained desired results.
     * *README.md* - description of the project 

## Batch Normalization

#### Target:
    No of parameters should be less than 50k.
    Accuracy of model should be between 70 and 94 which should be achieved in 20 epochs.

#### Result:
    No of parameters of model: 48512
    Test Accuracy: 77.8%
    Trained for 20 epochs and used less than 50k params.

#### Analysis:

    Model is overfitting.
    There was faster convergence with BN as it was applied on every layer.
    No of paremeters introduced through BN layer are less than Layer Normalization.
    Batch Normalization is used on every layer other than prediction layer.
    I see it misclassified many images as cat.

#### Graphs:
<img width="937" alt="image" src="https://github.com/das91t70/S8/assets/161017685/ad7cac12-f4c1-45be-95bb-ef9c29fd43ac">

#### 10 Misclassified images:
<img width="1222" alt="image" src="https://github.com/das91t70/S8/assets/161017685/6640a37b-01ba-481f-a1e9-ecc750ec519f">

## Layer Normalization

#### Target:
    No of parameters should be less than 50k.
    Accuracy of model should be between 70 and 94 which should be achieved in 20 epochs.

#### Result:
    No of parameters of model: 50900
    Best Test Accuracy: 69.33% ( 19th epoch)

#### Analysis:

    Model didn't overfit in initial epochs but it started overfitting in later epochs.
    I initially added Layer normalization to every layer by using which i was able to achieve accuracy upto 53%. In that case model was not overfitting. 
    As part of trial and error, i started removing layer normalization layers in few conv blocks which led to 69% accuracy.
    Training Loss was decreasing till 7 epochs and later at 11th epoch there was sudden increase in the training loss.
    No of parameters introduced through LN layer are greater than Batch Normalization.
    I see that Layer normalization may not be good fit for images as it was making model more heavier and also unable to understand basic features.
    I see it misclassified many images as bird and i feel it need to learn more patterns on identifying deer and house as it is still treating many as bird.

#### Graphs:

<img width="830" alt="image" src="https://github.com/das91t70/S8/assets/161017685/767b9981-a52d-49de-84bd-46ec6b5041cd">


#### 10 Misclassfied images:

<img width="1216" alt="image" src="https://github.com/das91t70/S8/assets/161017685/e2da7775-ddc6-423a-8345-39b5cd0f3fa2">


## Group Normalization

#### Target:
    No of parameters should be less than 50k.
    Accuracy of model should be between 70 and 94 which should be achieved in 20 epochs.

#### Result:
    No of parameters of model: 18212
    Best Test Accuracy: 68.3% ( 18th epoch)

#### Analysis:

    Model didn't overfit in initial epochs but it started overfitting in later epochs.
    I have taken same architecture used for Layer normalization and replaced layers with group normalization.
    As part of this network, I see that group normalization added less number of parameters when compared to Layer normalization.
    I see that this GN neural network can be trained few epochs / additional kernels can be added ( capacity can be increased) to gain more deeper patterns which would have helped model to identify differences between (airplane and bird ) and ( horse and deer ).

#### Graphs:

<img width="824" alt="image" src="https://github.com/das91t70/S8/assets/161017685/8836b452-e405-46d7-b974-8d66aa79e161">


#### 10 Misclassfied images:

<img width="1216" alt="image" src="https://github.com/das91t70/S8/assets/161017685/4df92ac7-596d-4e03-89be-a127f3ccb353">



