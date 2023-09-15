# %% (setup)
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
import einops
from dataclasses import dataclass
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from typing import List, Tuple, Dict, Type
from PIL import Image
from IPython.display import display
from pathlib import Path
import torchinfo
import json
import pandas as pd
from jaxtyping import Float, Int
# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_resnets"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from part2_cnns.solutions import Linear, Conv2d, Flatten, ReLU, MaxPool2d
from part3_resnets.utils import print_param_count
import part3_resnets.tests as tests
from plotly_utils import line, plot_train_loss_and_test_accuracy_from_trainer

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

MAIN = __name__ == "__main__"
# %% === BUILDING AND TRAINING A CNN ===
# %% define ConvNet
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, (3, 3), 1, 1)
        self.mp = MaxPool2d((2, 2), 2, 0)
        self.conv2 = Conv2d(32, 64, (3, 3), 1, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = ConvNet()
summary = torchinfo.summary(model, input_size=(1, 1, 28, 28))
print(summary)


# %% (get_mnist)
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)
# %% (define ConvNetTrainer)
@dataclass
class ConvNetTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = ConvNetTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    epochs: int = 3
    optimizer: Type[t.optim.Optimizer] = t.optim.Adam
    learning_rate: float = 1e-3
    subset: int = 10


class ConvNetTrainer:
    def __init__(self, args: ConvNetTrainingArgs):
        self.args = args
        self.model = ConvNet().to(device)
        self.optimizer = args.optimizer(self.model.parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_mnist(subset=args.subset)
        self.logged_variables = {"loss": [], "accuracy": []}

    def training_step(self, imgs: Tensor, labels: Tensor):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.logged_variables["loss"].append(loss.item())
        return loss

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)

    def train(self):
        progress_bar = tqdm(total=self.args.epochs * len(self.trainset) // self.args.batch_size)
        for epoch in range(self.args.epochs):
            for imgs, labels in self.train_dataloader():
                loss = self.training_step(imgs, labels)
                desc = f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}"
                progress_bar.set_description(desc)
                progress_bar.update()
            for imgs, labels in self.val_dataloader():
                loss = self.validation_step(imgs, labels)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)

    @t.inference_mode()
    def validation_step(self, imgs: Tensor, labels: Tensor):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = self.model(imgs)
        accuracy = t.sum(logits.argmax(dim=1) == labels) / len(labels)
        self.logged_variables["accuracy"].append(accuracy)


# %% train ConvNet
args = ConvNetTrainingArgs(batch_size=128)
trainer = ConvNetTrainer(args)
trainer.train()
line(
    trainer.logged_variables["loss"],
    yaxis_range=[0, max(trainer.logged_variables["loss"]) + 0.1],
    labels={"x": "Num batches seen", "y": "Cross entropy loss"},
    title="ConvNet training on MNIST",
    width=700
)
line(
    trainer.logged_variables["accuracy"],
    yaxis_range=[0, max(trainer.logged_variables["accuracy"]) + 0.1],
    labels={"x": "Epoch", "y": "Validation Accuracy"},
    title="ConvNet training on MNIST",
    width=700
)


# %% === ASSEMBLING RESNET ===
# %% (Sequential)
class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules) # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules) # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x


# %% implement BatchNorm2d
class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()

        self.register_buffer('running_mean', t.zeros(num_features))
        self.register_buffer('running_var', t.ones(num_features))
        self.register_buffer('num_batches_tracked', t.tensor(0))
        self.num_features = num_features

        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        b, _, h, w = x.shape
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = einops.repeat(self.running_mean, 'c -> b c h w', b=b, h=h, w=w)
            var = einops.repeat(self.running_var, 'c -> b c h w', b=b, h=h, w=w)
        x_normed = (x - mean) / t.sqrt(var + self.eps)
        return einops.einsum(self.weight, x_normed, 'c, b c h w -> b c h w') \
            + einops.repeat(self.bias, 'c -> b c h w', b=b, h=h, w=w)

    def extra_repr(self) -> str:
        pass


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)


# %% implement AveragePool
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return einops.reduce(x, 'b c h w -> b c', 'mean')

# %% implement ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.main_path = Sequential(
            Conv2d(in_feats, out_feats, 3, first_stride, 1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, 3, 1, 1),
            BatchNorm2d(out_feats),
        )
        self.relu = ReLU()
        if(first_stride > 1):
            self.optional_path = Sequential(
                Conv2d(in_feats, out_feats, 1, first_stride, 0),
                BatchNorm2d(out_feats)
            )
        else:
            self.optional_path = nn.Identity()

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        return self.relu(self.main_path(x) + self.optional_path(x))




# %% implement BlockGroup
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        self.residual_blocks = Sequential(
            ResidualBlock(in_feats, out_feats, first_stride),
            *[ResidualBlock(out_feats, out_feats, 1) for _ in range(n_blocks-1)],
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.residual_blocks(x)


# %% implement ResNet34
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        conv_feats = 64
        self.conv1 = Conv2d(3, conv_feats, 7, 2, 3)
        self.batch_norm = BatchNorm2d(conv_feats)
        self.relu = ReLU()
        self.mp = MaxPool2d(3, 2, 1)

        in_features_per_group = [conv_feats, *out_features_per_group[:-1]]
        self.block_groups = nn.Sequential(*(
            BlockGroup(n_blocks, in_feats, out_feats, first_stride) for
            (n_blocks, in_feats, out_feats, first_stride) in
            zip(n_blocks_per_group, in_features_per_group, out_features_per_group, first_strides_per_group)
        ))
        self.average_pool = AveragePool()
        self.flatten = Flatten()
        self.fc1 = Linear(out_features_per_group[-1], n_classes)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        x = self.mp(self.relu(self.batch_norm(self.conv1(x))))
        x = self.block_groups(x)
        x = self.flatten(self.average_pool(x))
        return self.fc1(x)


my_resnet = ResNet34()


# %% (copy_weights)
def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
my_resnet = copy_weights(my_resnet, pretrained_resnet)


# %% (get images)
IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# %% prepare_data
def prepare_data(images: List[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    return t.stack([IMAGENET_TRANSFORM(image) for image in images])


prepared_images = prepare_data(images)

assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

# %% (predict)
def predict(model, images):
    logits: t.Tensor = model(images)
    return logits.argmax(dim=1)

# %% (output predictions)
with open(section_dir / "imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# Check your predictions match the pretrained model's
my_predictions = predict(my_resnet, prepared_images)
pretrained_predictions = predict(pretrained_resnet, prepared_images)
assert all(my_predictions == pretrained_predictions)

# Print out your predictions, next to the corresponding images
for img, label in zip(images, my_predictions):
    print(f"Class {label}: {imagenet_labels[label]}")
    display(img)
    print()


# %% == RESNET FEATURE EXTRACTION (FINETUNING RESNET) ==
# %% prepare ResNet for feature extraction
def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
    '''
    Creates a ResNet34 instance, replaces its final linear layer with a classifier
    for `n_classes` classes, and freezes all weights except the ones in this layer.

    Returns the ResNet model.
    '''
    my_resnet = ResNet34()
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    copy_weights(my_resnet, pretrained_resnet)

    my_resnet.requires_grad_(False)
    my_resnet.fc1 = Linear(my_resnet.fc1.in_features, n_classes)
    return my_resnet


tests.test_get_resnet_for_feature_extraction(get_resnet_for_feature_extraction)


# %% (get cifar)
def get_cifar(subset: int):
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)

    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))

    return cifar_trainset, cifar_testset


@dataclass
class ResNetTrainingArgs():
    batch_size: int = 64
    epochs: int = 3
    optimizer: Type[t.optim.Optimizer] = t.optim.Adam
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 10


# %% write training loop for ResNet feature extraction 
class ResNetTrainer(ConvNetTrainer):
    def __init__(self, args: ResNetTrainingArgs):
        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        self.optimizer = args.optimizer(self.model.parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=args.subset)
        self.logged_variables = {"loss": [], "accuracy": []}

args = ResNetTrainingArgs()
trainer = ResNetTrainer(args)
trainer.train()
plot_train_loss_and_test_accuracy_from_trainer(trainer, title="Feature extraction with ResNet34")
# %%
