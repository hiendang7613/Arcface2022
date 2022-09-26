import tensorflow as tf
import argparse
from Network.model import MyModel
from Network.head.archead import ArcHead
from Tensorflow.TFRecord.tfrecord import TFRecordData
from training_supervisor import TrainingSupervisor
from LossFunction.losses import arcface_loss_v4
from evalute import EvaluteObjects
from config import train_config, mlflow_config
from utlis.utlis import set_env_vars


def train():
    print("Options-Training:")
    for k, v in locals().items():
        print(f"  {k}: {v}")

    # get hyper-parameter
    tfrecord_file = train_config.tfrecord_file
    tfrecord_file_eval = train_config.tfrecord_file_eval
    file_pair_eval = train_config.file_pair_eval
    num_classes = train_config.num_classes
    num_images = train_config.num_images
    embedding_size = train_config.embedding_size
    batch_size = train_config.batch_size
    epochs = train_config.epochs
    input_shape = train_config.input_shape
    training_dir = train_config.training_dir
    export_dir = train_config.export_dir

    # 3.6 define the learning rate schedule
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    p = int(512.0/batch_size)
    lr_steps = [p*val for val in [40000, 60000, 80000]]
    print(lr_steps)
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001], name='lr_schedule')  # type: ignore

    # chosing model
    type_backbone = 'hiendv_model'
    archead = ArcHead(num_classes=num_classes)
    model = MyModel(type_backbone=type_backbone,
                    input_shape=input_shape,
                    embedding_size=embedding_size,
                    header=archead)
    model.build(input_shape=(None, input_shape, input_shape, 3))
    optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True, epsilon=0.001)
    model.summary()

    # init loss function
    loss_fn = arcface_loss_v4(margin=0.5, scale=64, n_classes=num_classes)

    # dataloader
    dataloader_train = TFRecordData.load(record_name=tfrecord_file,
                                         shuffle=True,
                                         batch_size=batch_size,
                                         is_repeat=False,
                                         binary_img=True,
                                         is_crop=True,
                                         reprocess=False,
                                         num_classes=num_classes,
                                         buffer_size=2048)

    supervisor = TrainingSupervisor(train_dataloader=dataloader_train,
                                    validation_dataloader=None,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    model=model,
                                    save_freq=1000,
                                    monitor='categorical_accuracy',
                                    mode='max',
                                    training_dir=training_dir,
                                    name='Trainer_Supervisor')

    supervisor.restore(weights_only=False, from_scout=True)
    supervisor.train(epochs=epochs, steps_per_epoch=num_images // batch_size)
    supervisor.export(model=model.backbone, export_dir=export_dir)
    supervisor.mlflow_artifact(model=model.backbone,
                               tensorboard_dir=training_dir,
                               export_dir=export_dir)

    # evaluate ...
    eval_class = EvaluteObjects(tfrecord_file=tfrecord_file_eval,
                                file_pairs=file_pair_eval,
                                batch_size=batch_size)
    metrics = eval_class.activate(model=model.backbone, embedding_size=embedding_size)
    eval_class.mlflow_logs(dict_metrics=metrics)


def mlflow_run():
    args_mlflow = parser_mlflow()
    print("Options-Mlflow:")
    for k, v in args_mlflow.items():
        print(f"  {k}: {v}")

    args_mlflow['model_name'] = None if not args_mlflow['model_name'] or args_mlflow['model_name'] == "None" else \
        args_mlflow['model_name']

    if args_mlflow['tensorflow_autolog']:
        mlflow.tensorflow.autolog()
    if args_mlflow['mlflow_autolog']:
        mlflow.autolog()

    try:
        exp_id = mlflow.create_experiment(name=args_mlflow['experiment_name'])
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(name=args_mlflow['experiment_name']).experiment_id

    with mlflow.start_run(experiment_id=exp_id, run_name=args_mlflow['model_name']) as run:
        print("MLflow:")
        print("  run_id:", run.info.run_id)
        print("  experiment_id:", run.info.experiment_id)
        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.tensorflow", tf.__version__)
        mlflow.set_tag("mlflow_autolog", args_mlflow['mlflow_autolog'])
        mlflow.set_tag("tensorflow_autolog", args_mlflow['tensorflow_autolog'])
        mlflow.set_tag("mlflow_custom_log", args_mlflow['mlflow_custom_log'])
        mlflow.set_tag("Type of model", args_mlflow['model_name'])
        mlflow.set_tag("Developer", args_mlflow['user_name'])
        train(run, args_mlflow['model_name'], args_mlflow['mlflow_custom_log'])


if __name__ == '__main__':
    # set_env_vars()
    # mlflow_run()
    train()

