"""
USAGE

# training with Faster RCNN ResNet50 FPN model without mosaic or any other augmentation:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --data data_configs/voc.yaml --mosaic 0 --batch 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default):
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4

# Training on ResNet50 FPN with custom project folder name with mosaic augmentation (ON by default) and added training augmentations:
python train.py --model fasterrcnn_resnet50_fpn --epochs 2 --use-train-aug --data data_configs/voc.yaml --name resnet50fpn_voc --batch 4

# Distributed training:
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --data data_configs/smoke.yaml --epochs 100 --model fasterrcnn_resnet50_fpn --name smoke_training --batch 16
"""
from torch_utils.engine import (
    train_one_epoch, evaluate, utils
)
from torch.utils.data import (
    distributed, RandomSampler, SequentialSampler, Subset
)
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from utils.general import (
    set_training_dir, Averager, 
    save_model, save_loss_plot,
    show_tranformed_image,
    save_mAP, save_model_state, SaveBestModel,
    yaml_save, init_seeds
)
from utils.logging import (
    set_log, coco_log,
    set_summary_writer, 
    tensorboard_loss_log, 
    tensorboard_map_log,
    csv_log,
    wandb_log, 
    wandb_save_model,
    wandb_init
)

import torch
import argparse
import yaml
import numpy as np
import torchinfo
import os
from sklearn.model_selection import KFold

torch.multiprocessing.set_sharing_strategy('file_system')

RANK = int(os.getenv('RANK', -1))

# For same annotation colors each time.
np.random.seed(42)

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn_v2',
        help='name of the model'
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default='cuda',
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs', 
        default=5,
        type=int,
        help='number of epochs to train for'
    )
    parser.add_argument(
        '-j', '--workers', 
        default=4,
        type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=4, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '--lr', 
        default=0.001,
        help='learning rate for the optimizer',
        type=float
    )
    parser.add_argument(
        '-ims', '--imgsz',
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-n', '--name', 
        default=None, 
        type=str, 
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '-vt', '--vis-transformed', 
        dest='vis_transformed', 
        action='store_true',
        help='visualize transformed images fed to the network'
    )
    parser.add_argument(
        '--mosaic', 
        default=0.0,
        type=float,
        help='probability of applying mosaic, (default, always apply)'
    )
    parser.add_argument(
        '-uta', '--use-train-aug', 
        dest='use_train_aug', 
        action='store_true',
        help='whether to use train augmentation, blur, gray, \
              brightness contrast, color jitter, random gamma \
              all at once'
    )
    parser.add_argument(
        '-ca', '--cosine-annealing', 
        dest='cosine_annealing', 
        action='store_true',
        help='use cosine annealing warm restarts'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None, 
        type=str,
        help='path to model weights if using pretrained weights'
    )
    parser.add_argument(
        '-r', '--resume-training', 
        dest='resume_training', 
        action='store_true',
        help='whether to resume training, if true, \
            loads previous training plots and epochs \
            and also loads the otpimizer state dictionary'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    parser.add_argument(
        '--world-size', 
        default=1, 
        type=int, 
        help='number of distributed processes'
    )
    parser.add_argument(
        '--dist-url',
        default='env://',
        type=str,
        help='url used to set up the distributed training'
    )
    parser.add_argument(
        '-dw', '--disable-wandb',
        dest="disable_wandb",
        action='store_true',
        help='whether to use the wandb'
    )
    parser.add_argument(
        '--sync-bn',
        dest='sync_bn',
        help='use sync batch norm',
        action='store_true'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='use automatic mixed precision'
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=int ,
        help='golabl seed for training'
    )
    parser.add_argument(
        '--project-dir',
        dest='project_dir',
        default=None,
        help='save resutls to custom dir instead of `outputs` directory, \
              --project-dir will be named if not already present',
        type=str
    )
    parser.add_argument(
        '--kfold',
        default=5,
        type=int,
        help='number of folds for KFold cross-validation'
    )

    args = vars(parser.parse_args())
    return args

def main(args):
    # Initialize distributed mode.
    utils.init_distributed_mode(args)

    # Initialize W&B with project name.
    if not args['disable_wandb']:
        wandb_init(name=args['name'])
    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    init_seeds(args['seed'] + 1 + RANK, deterministic=True)
    
    # Settings/parameters/constants.
    TRAIN_DIR_IMAGES = os.path.normpath(data_configs['TRAIN_DIR_IMAGES'])
    TRAIN_DIR_LABELS = os.path.normpath(data_configs['TRAIN_DIR_LABELS'])
    VALID_DIR_IMAGES = os.path.normpath(data_configs['VALID_DIR_IMAGES'])
    VALID_DIR_LABELS = os.path.normpath(data_configs['VALID_DIR_LABELS'])
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = args['workers']
    DEVICE = torch.device(args['device'])
    print("device",DEVICE)
    NUM_EPOCHS = args['epochs']
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']
    BATCH_SIZE = args['batch']
    VISUALIZE_TRANSFORMED_IMAGES = args['vis_transformed']
    OUT_DIR = set_training_dir(args['name'], args['project_dir'])
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    SCALER = torch.cuda.amp.GradScaler() if args['amp'] else None
    # Set logging file.
    set_log(OUT_DIR)
    writer = set_summary_writer(OUT_DIR)

    yaml_save(file_path=os.path.join(OUT_DIR, 'opt.yaml'), data=args)

    # Model configurations
    IMAGE_SIZE = args['imgsz']
    
    dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, 
        TRAIN_DIR_LABELS,
        IMAGE_SIZE, 
        CLASSES,
        use_train_aug=args['use_train_aug'],
        mosaic=args['mosaic'],
        square_training=args['square_training']
    )
    
    kfold = KFold(n_splits=args['kfold'], shuffle=True, random_state=args['seed'])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        if args['distributed']:
            train_sampler = distributed.DistributedSampler(train_subset)
            valid_sampler = distributed.DistributedSampler(val_subset, shuffle=False)
        else:
            train_sampler = RandomSampler(train_subset)
            valid_sampler = SequentialSampler(val_subset)

        train_loader = create_train_loader(
            train_subset, BATCH_SIZE, NUM_WORKERS, batch_sampler=train_sampler
        )
        valid_loader = create_valid_loader(
            val_subset, BATCH_SIZE, NUM_WORKERS, batch_sampler=valid_sampler
        )
        
        print(f"Number of training samples: {len(train_subset)}")
        print(f"Number of validation samples: {len(val_subset)}\n")

        if VISUALIZE_TRANSFORMED_IMAGES:
            show_tranformed_image(train_loader, DEVICE, CLASSES, COLORS)

        # Initialize the Averager class.
        train_loss_hist = Averager()
        train_loss_list = []
        loss_cls_list = []
        loss_box_reg_list = []
        loss_objectness_list = []
        loss_rpn_list = []
        train_loss_list_epoch = []
        val_map_05 = []
        val_map = []
        best_map = 0

        if not args['disable_wandb']:
            wandb.watch(create_model(args['model'], NUM_CLASSES))

        # Initialize the model and move to the computation device.
        model = create_model(args['model'], NUM_CLASSES)
        model.to(DEVICE)

        if args['weights'] is not None:
            weights_dict = torch.load(args['weights'], map_location=DEVICE)
            model.load_state_dict(weights_dict['model_state_dict'])

        # Get the model summary.
        torchinfo.summary(model, (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

        # Get the parameters.
        params = [p for p in model.parameters() if p.requires_grad]
        
        # Optimizer.
        optimizer = torch.optim.AdamW(params, lr=args['lr'])
        if args['resume_training'] and args['weights'] is not None:
            optimizer.load_state_dict(weights_dict['optimizer_state_dict'])
            init_epoch = weights_dict['epoch']
        else:
            init_epoch = 1

        # Learning rate scheduler.
        if args['cosine_annealing']:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1, verbose=True
            )
        else:
            lr_scheduler = None

        save_best_model = SaveBestModel()

        # Training loop.
        for epoch in range(init_epoch, NUM_EPOCHS + 1):
            # Train for one epoch.
            train_loss_hist.reset()
            _, batch_loss_list, batch_loss_cls_list, \
                batch_loss_box_reg_list, batch_loss_objectness_list, \
                batch_loss_rpn_list = train_one_epoch(
                model, train_loader, optimizer,
                DEVICE, epoch, train_loss_hist,
                args['cosine_annealing'], lr_scheduler, SCALER, args['amp']
            )

            loss_cls_list.extend(batch_loss_cls_list)
            loss_box_reg_list.extend(batch_loss_box_reg_list)
            loss_objectness_list.extend(batch_loss_objectness_list)
            loss_rpn_list.extend(batch_loss_rpn_list)

            train_loss_list_epoch.append(train_loss_hist.value)
            train_loss_list.extend(batch_loss_list)
            print(f"Epoch #{epoch} loss: {train_loss_hist.value}")

            coco_evaluator, stats, val_pred_images = evaluate(
                model, valid_loader, DEVICE, args['epochs'], epoch, args['amp']
            )
            _, mAP_0_5, mAP = stats
            val_map_05.append(mAP_0_5)
            val_map.append(mAP)

            if not args['disable_wandb']:
                wandb_log({
                    'train_loss': train_loss_hist.value,
                    'val_mAP_0.5': mAP_0_5,
                    'val_mAP': mAP
                })

            if args['cosine_annealing']:
                print('Cosine annealing warm restarts lr: {:.6f}'.format(optimizer.param_groups[0]["lr"]))

            # Save the current epoch model state
            save_model_state(
                model, optimizer, args['model'], args['epochs'], epoch, OUT_DIR
            )

            # Save model if a better mAP is obtained.
            save_best_model(
                mAP, epoch, model, optimizer, args['model'],
                args['epochs'], OUT_DIR
            )
        
        print(f"Finished fold {fold + 1}")

    if not args['disable_wandb']:
        wandb.finish()
    writer.close()

if __name__ == '__main__':
    args = parse_opt()
    main(args)

