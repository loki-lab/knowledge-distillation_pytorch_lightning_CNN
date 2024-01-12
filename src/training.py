from pytorch_lightning import Trainer
from lightning_model import LightningModel
from models import VGG11
from utils import get_transforms, load_data
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
import comet_ml

path_datasets = "datasets/PetImg"
in_chanel = 3
num_classes = 2

train_size = 20000
val_size = 5000

transforms = get_transforms()["train"]


if __name__ == "__main__":
    # load data
    train_ds, val_ds = load_data(path_data=path_datasets,
                                 transform=transforms,
                                 train_size=train_size,
                                 val_size=val_size)

    # load model
    vgg11_model = VGG11(in_channels=in_chanel, num_classes=num_classes)
    vgg11_lightning_model = LightningModel(model=vgg11_model, train_ds=train_ds, val_ds=val_ds)

    # callbacks for training
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints",
                                          filename="vgg11_best_model",
                                          save_top_k=1,
                                          save_last=True,
                                          monitor="val_acc",
                                          mode="max")

    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=5,
                                        verbose=True)

    # visualization training
    comet_ml.init(project_name="comet_training-student-pytorch-lightning")
    comet_logger = CometLogger(api_key="LBJ57ChbNyjtlvNf4wyWtrJnH")
    comet_logger.log_hyperparams({"batch_size": 32})

    # training model
    trainer = Trainer(accelerator="gpu",
                      devices="auto",
                      max_epochs=30,
                      logger=comet_logger,
                      callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(vgg11_lightning_model)
