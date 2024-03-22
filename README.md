## S8 Assignment

Part 1

* Batch Normalization is applied
* Batch Normalization is used on every layer other than prediction layer.
* No of parameters of model: 48512
* Test Accuracy: 77.8%
* Trained for 20 epochs and used less than 50k params

#### Analysis:

* Model is overfitting
* There was increase in learning rate and faster convergence with BN
* We have not only applied image normalization on the image but also applied batch normalization on every layer.
