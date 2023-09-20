# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
In this project, I have decided to fine tune a Resnet101 image classification model. That being said, I tuned the following hyperparameters:

**Learning Rate**: The learning rate determines the step size at which the model adjusts its internal parameters during training. A high learning rate may cause the model to converge quickly but risk overshooting the optimal solution. Conversely, a low learning rate may result in slow convergence or getting stuck in suboptimal solutions. Finding an appropriate learning rate is essential for achieving faster convergence and better generalization .
<br><br>
**Batch Size**: The batch size refers to the number of training examples used in each iteration of gradient descent. A larger batch size can lead to faster training as more examples are processed simultaneously, but it requires more memory. On the other hand, a smaller batch size can provide a noisier estimate of the gradient but may allow for better generalization. Selecting an appropriate batch size depends on factors such as available computational resources, dataset size, and model complexity .
<br><br>
**Number of Epochs**: An epoch represents a complete pass through the entire training dataset. Training for too few epochs may result in underfitting, where the model fails to capture complex patterns in the data. Conversely, training for too many epochs can lead to overfitting, where the model becomes too specialized to the training data and performs poorly on unseen examples. Determining the optimal number of epochs involves finding a balance between underfitting and overfitting by monitoring performance on a validation set .
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search.

<h1 align="center">Hyperparameter tunning code</h1>

![completed_hyperparameter_tunning](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/blob/main/images/hyperparameter_tunning.png)

<br>

<h1 align="center">total training runs</h1>

![total_training runs](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/blob/main/images/training_runs.png)

<br>

<h1 align="center">Log metrics during training can be monitored with cloudwatch</h1>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/8d35351d-1ce9-44b3-b136-d831713d9752)

<br>

<h1 align="center">Best Hyperparameters found</h1>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/303877fa-323c-48f3-965d-087e97fdb304)

<br>
## Debugging and Profiling
In order to perform debugging and Profiling, I used sagemaker debugger and profiler tools. In practice, you need to modify your estimator definition, by adding debugging and profilling configurations. Besides that, you also need to add  debugger hooks in your training script, 
so you can monitor the metrics you are interested in.

<h3 align="center">Defining profiler and debug rules</h3>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/66e1f1a9-ad07-4d35-85b6-f1b71eb9eaf2)

<br>
<h3 align="center">modifying estimator</h3>

![Screenshot from 2023-09-20 16-52-11](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/6ff8de51-9105-4ece-8cb8-412c94370f12)

<br>
<h3 align="center">modifying training script</h3>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/0337fb42-d6b5-449f-98ff-7fbbb515f90d)


### Results
For my particular experiment, I chose to monitor the loss, weights, and biases during the training process. The respective tensors are sampled and saved in a S3 bucket for further analysis. You can also programmatically create a trial and access the outputs:
<br>
<h3 align="center">Sagemaker debug results</h3>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/30699f4f-5b09-46dc-b5bd-e717abc60ff9)

Then you can see, for instance, your train and validation loss curves:
<br><br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/c687d010-334e-43a6-b3ff-2fbf6241dfe9)

Besides that, sagemaker profiler generates a HTML report for you, analyzing your training jobs according to several different metrics:
<br><br>
![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/77a33927-8011-4999-91b2-8b2fa88b5bdd)

## What are the results/insights did you get by profiling/debugging your model?

Fortunately, my model didn't flag any alert, and the loss curve is not too noisy. But taking a look at CPU utilization metrics, It seems the machine that
I chose to use (ml.c5.2xlarge) is underutilized because it rarely passes 37% CPU utilization. Hence If in the future I were to build a retraining pipeline or simply repeat the experiment, I could choose a smaller instance and incur fewer costs.

<h3 align="center">training instance metrics</h3>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/5e39ca92-f9ed-4f0b-8c4a-9586c88d4e69)

<br>

## Model Deployment
The model that was fine-tuned in the previous step was deployed to a sagemaker endpoint. In summary, you need to create an inference script, that will be run in the endpoint instance and will be responsible for running predictions against incoming data. Besides that, it is important to define a predictor, which will serialize the data to be sent to the endpoint and will deserialize the received responses.

<h3 align="center">model deployment code</h3>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/49fcf9b2-9464-49e3-8e96-a201364910f8)

<h3 align="center">succesfully deployed endpoint</h3>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/blob/main/images/active_endpoints.png)

<h3 align="center">querying endpoint</h3>

![image](https://github.com/hualcosa/AWS-Machine-Learning-Engineer-NanoDegree-Project3/assets/46836901/bb89d325-b22c-4eb6-b865-4fdbe4cb84ba)
