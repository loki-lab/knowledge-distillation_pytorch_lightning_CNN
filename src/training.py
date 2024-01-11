from pytorch_lightning import Trainer
from lightning_model import LightningModel
from torchvision.datasets import ImageFolder
from models import VGG16
from torchvision import transforms
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
import comet_ml

path_datasets = "datasets/PetImg"
in_chanel = 3
num_classes = 2

train_size = 20000
val_size = 5000

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((224, 224), antialias=True),
                                 transforms.Normalize((0.5,), (0.5,)),
                                 transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
                                 transforms.RandomRotation(degrees=(0, 180)),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.RandomVerticalFlip(p=0.5)])


def load_data(path_data):
    dataset = ImageFolder(path_data, transforms)
    train_data, val_data = random_split(dataset, [train_size, val_size])
    return train_data, val_data


if __name__ == "__main__":
    # load data
    train_ds, val_ds = load_data(path_data=path_datasets)

    # load model
    vgg16_model = VGG16(in_chanel=in_chanel, num_classes=num_classes)
    vgg16_lightning_model = LightningModel(model=vgg16_model, train_ds=train_ds, val_ds=val_ds)

    # callbacks for training
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints",
                                          save_top_k=-1,
                                          monitor="val_acc_step",
                                          mode="max")

    early_stop_callback = EarlyStopping(monitor="val_acc_step",
                                        min_delta=0.05,
                                        mode="max",
                                        patience=5)

    # visualization training
    comet_ml.init(project_name="comet-teacher-pytorch-lightning")
    comet_logger = CometLogger(api_key="LBJ57ChbNyjtlvNf4wyWtrJnH")
    comet_logger.log_hyperparams({"batch_size": 32, "learning_rate": 0.0005})

    # training model
    trainer = Trainer(accelerator="gpu",
                      devices="auto",
                      max_epochs=30,
                      logger=comet_logger,
                      callbacks=[checkpoint_callback, early_stop_callback])

    trainer.fit(vgg16_lightning_model)
