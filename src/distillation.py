from pytorch_lightning import Trainer
from lightning_model import LightningModelDistill, LightningModel
from models import VGG16, VGG11
from utils import get_transforms, load_data
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
import comet_ml

path_datasets = "datasets/PetImg"
checkpoint = "checkpoints/"
in_chanel = 3
num_classes = 2

train_size = 20000
val_size = 5000

temp = 2
alpha = 0.8

transform = get_transforms()["train"]

if __name__ == "__main__":
    # load data
    train_ds, val_ds = load_data(path_data=path_datasets,
                                 transform=transform,
                                 train_size=train_size,
                                 val_size=val_size)

    # load model
    vgg16_model = VGG16(in_chanels=in_chanel, num_classes=num_classes)
    vgg11_model = VGG11(in_channels=in_chanel, num_classes=num_classes)

    vgg16_lightning_model = LightningModel(model=vgg16_model).load_from_checkpoint(checkpoint)
    distillation_lightning_model = LightningModelDistill(teacher_model=vgg16_model,
                                                         student_model=vgg11_model,
                                                         train_ds=train_ds,
                                                         val_ds=val_ds,
                                                         temp=temp,
                                                         alpha=alpha
                                                         )

    # callbacks for training
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints",
                                          save_top_k=4,
                                          monitor="val_acc_step",
                                          mode="max")

    early_stop_callback = EarlyStopping(monitor="val_loss_step",
                                        mode="min",
                                        patience=4,
                                        verbose=True)

    # visualization training
    comet_ml.init(project_name="comet_training-distillation-pytorch-lightning")
    comet_logger = CometLogger(api_key="LBJ57ChbNyjtlvNf4wyWtrJnH")
    comet_logger.log_hyperparams({"batch_size": 32})

    # training model
    trainer = Trainer(accelerator="gpu",
                      devices="auto",
                      max_epochs=30,
                      logger=comet_logger,
                      callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(distillation_lightning_model)
