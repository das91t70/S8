# S8 Assignment

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

#### 10 Misclassified images
    <img width="1222" alt="image" src="https://github.com/das91t70/S8/assets/161017685/6640a37b-01ba-481f-a1e9-ecc750ec519f">




