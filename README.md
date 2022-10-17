# Training & Tracking result 

## Dataset
````
 Train: https://drive.google.com/file/d/1gMnPK2NDDSHzNck73TsJ2XB1KmQ4evCp/view?usp=sharing
 Validate: LFW_Mask or MFR_2, Coming soon ...
 Test: Coming soon 
````

### Config file at ./config/train_config.py
you need to update some hyperparameter in there
````
 Important params:
  tfrecord_file: it using for training
  tfrecord_file_eval: it using for evaluate
  file_pair_eval: file pairs.txt      
  num_classes: the amount of person class
  num_images: the amount of images all class
````

### Config mlflow at ./config/mlflow_config.py
````
 user_name: who use it 
 run_name_model_type: type of model train 
 model_name: architecture name 
 experiment_name: experiment to tracking
````

## Training
> python train.py

## Evaluate
> python evaluate.py





