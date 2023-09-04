import argparse
import os

from loguru import logger
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from virtex.config import Config
from virtex.factories import (
    DownstreamDatasetFactory,
    PretrainingModelFactory,
    OptimizerFactory,
    LRSchedulerFactory,
)
from virtex.utils.checkpointing import CheckpointManager
from virtex.utils.common import common_parser, common_setup, cycle
import virtex.utils.distributed as dist
from virtex.utils.metrics import TopkAccuracy
from virtex.utils.timer import Timer


# fmt: off
parser = common_parser(
    description="""Do image classification with linear models and frozen
    feature extractor, or fine-tune the feature extractor end-to-end."""
)
group = parser.add_argument_group("Downstream config arguments.")
group.add_argument(
    "--down-config", metavar="FILE", help="Path to a downstream config file."
)
group.add_argument(
    "--down-config-override", nargs="*", default=[],
    help="A list of key-value pairs to modify downstream config params.",
)

parser.add_argument_group("Checkpointing and Logging")
parser.add_argument(
    "--weight-init", choices=["random", "imagenet", "torchvision", "virtex"],
    default="virtex", help="""How to initialize weights:
        1. 'random' initializes all weights randomly
        2. 'imagenet' initializes backbone weights from torchvision model zoo
        3. {'torchvision', 'virtex'} load state dict from --checkpoint-path
            - with 'torchvision', state dict would be from PyTorch's training
              script.
            - with 'virtex' it should be for our full pretrained model."""
)
parser.add_argument(
    "--log-every", type=int, default=50,
    help="""Log training curves to tensorboard after every these many iterations
    only master process logs averaged loss values across processes.""",
)
parser.add_argument(
    "--checkpoint-path",
    help="""Path to load checkpoint and run downstream task evaluation. The
    name of checkpoint file is required to be `model_*.pth`, where * is
    iteration number from which the checkpoint was serialized."""
)
parser.add_argument(
    "--checkpoint-every", type=int, default=5000,
    help="""Serialize model to a checkpoint after every these many iterations.
    For ImageNet, (5005 iterations = 1 epoch); for iNaturalist (1710 iterations
    = 1 epoch).""",
)
# fmt: on


def main(_A: argparse.Namespace):

    if _A.num_gpus_per_machine == 0:
        # Set device as CPU if num_gpus_per_machine = 0.
        device = torch.device("cpu")
    else:
        # Get the current device as set for current distributed process.
        # Check `launch` function in `virtex.utils.distributed` module.
        device = torch.cuda.current_device()

    # Create a downstream config object (this will be immutable) and perform
    # common setup such as logging and setting up serialization directory.
    _DOWNC = Config(_A.down_config, _A.down_config_override)
    common_setup(_DOWNC, _A, job_type="downstream")

    # Create a (pretraining) config object and backup in serializaion directory.
    _C = Config(_A.config, _A.config_override)
    _C.dump(os.path.join(_A.serialization_dir, "pretrain_config.yaml"))

    # Get dataset name for tensorboard logging.
    DATASET = _DOWNC.DATA.ROOT.split("/")[-1]

    # Set number of output classes according to dataset:
    NUM_CLASSES_MAPPING = {"imagenet": 1000, "inaturalist": 8142}
    NUM_CLASSES = NUM_CLASSES_MAPPING[DATASET]

    # Initialize model using pretraining config.
    pretrained_model = PretrainingModelFactory.from_config(_C)

    # Load weights according to the init method, do nothing for `random`, and
    # `imagenet` is already taken care of.
    if _A.weight_init == "virtex":
        CheckpointManager(model=pretrained_model).load(_A.checkpoint_path)
    elif _A.weight_init == "torchvision":
        # Keep strict=False because this state dict may have weights for
        # last fc layer.
        pretrained_model.visual.cnn.load_state_dict(
            torch.load(_A.checkpoint_path, map_location="cpu")["state_dict"],
            strict=False,
        )

    # Pull out the CNN (torchvision-like) from our pretrained model and add
    # back the FC layer - this is exists in torchvision models, and is set to
    # `nn.Identity()` during pretraining.
    model = pretrained_model.visual.cnn  # type: ignore
    model.fc = nn.Linear(_DOWNC.MODEL.VISUAL.FEATURE_SIZE, NUM_CLASSES).to(device)
    model = model.to(device)

    # Re-initialize the FC layer.
    torch.nn.init.normal_(model.fc.weight.data, mean=0.0, std=0.01)
    torch.nn.init.constant_(model.fc.bias.data, 0.0)

    dir = '/cluster/home/zhangzirui/bysj/models/resnet50_virtex_caption_final.pth'
    torch.save(
        model.state_dict(),
        dir
    )

    print("Save Model")

    del pretrained_model


if __name__ == "__main__":
    _A = parser.parse_args()

    # Add an arg in config override if `--weight-init` is imagenet.
    if _A.weight_init == "imagenet":
        _A.config_override.extend(["MODEL.VISUAL.PRETRAINED", True])

    main(_A)
