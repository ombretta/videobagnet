import argparse
from pathlib import Path
import shlex


def parse_opts(arguments_string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default=None,
                        type=Path,
                        help='Root directory path')
    parser.add_argument('--video_path',
                        default=None,
                        type=Path,
                        help='Directory path of videos')
    parser.add_argument('--annotation_path',
                        default=None,
                        type=Path,
                        help='Annotation file path')
    parser.add_argument('--result_path',
                        default=None,
                        type=Path,
                        help='Result directory path')
    parser.add_argument(
        '--save_opts',
        default=False,
        action='store_true',
        help='If true the opts are saved in opts.json.'
    )
    parser.add_argument(
        '--dataset',
        default='kinetics',
        type=str,
        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51 | breakfast | movingmnist)')
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51, breakfast/movingmnist: 10)'
    )
    parser.add_argument(
        '--loss_weights',
        default=None,
        type=float,
        nargs='+',
        help='Class weights for the loss function. Default: None.'
    )
    parser.add_argument(
        '--multilabel',
        action='store_true',
        help='If true the model is adapted for multi-label classification problem.'
    )
    parser.add_argument('--n_pretrain_classes',
                        default=0,
                        type=int,
                        help=('Number of classes of pretraining task.'
                              'When using --pretrain_path, this must be set.'))
    parser.add_argument('--pretrain_path',
                        default=None,
                        type=Path,
                        help='Pretrained model path (.pth).')
    parser.add_argument(
        '--ft_begin_module',
        default='',
        type=str,
        help=('Module name of beginning of fine-tuning'
              '(conv1, layer1, fc, denseblock1, classifier, ...).'
              'The default means all layers are fine-tuned.'))
    parser.add_argument('--sample_size',
                        default=64,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_duration',
                        default=48,
                        type=int,
                        help='Temporal duration of inputs')
    parser.add_argument(
        '--sample_t_stride',
        default=1,
        type=int,
        help='If larger than 1, input frames are subsampled with the stride.')
    parser.add_argument(
        '--n_segments',
        default=1,
        type=int,
        help='Parameters used for the segment classifier. Defines the number \
            of segments processed in parallel.')
    parser.add_argument(
        '--train_crop',
        default='random',
        type=str,
        help=('Spatial cropping method in training. '
              'random is uniform. '
              'corner is selection from 4 corners and 1 center. '
              '(random | corner | center)'))
    parser.add_argument('--train_crop_min_scale',
                        default=0.25,
                        type=float,
                        help='Min scale for random cropping in training')
    parser.add_argument('--train_crop_min_ratio',
                        default=0.75,
                        type=float,
                        help='Min aspect ratio for random cropping in training')
    parser.add_argument('--no_hflip',
                        action='store_true',
                        help='If true holizontal flipping is not performed.')
    parser.add_argument('--colorjitter',
                        action='store_true',
                        help='If true colorjitter is performed.')
    parser.add_argument('--gaussiannoise',
                        action='store_true',
                        help='If true independent gaussian noise added to the frames.')
    parser.add_argument('--gaussiannoise_at_inference',
                        action='store_true',
                        help='If true independent gaussian noise added to the frames at test time.')
    parser.add_argument('--gaussiannoise_mean',
                        default=0.0,
                        type=float,
                        help='Mean of Gaussian noise.')
    parser.add_argument('--gaussiannoise_std',
                        default=1.0,
                        type=float,
                        help='Standard deviation of Gaussian noise.')
    parser.add_argument('--gaussianblur',
                        action='store_true',
                        help='If true gaussian blur is performed frame by frame.')
    parser.add_argument('--gaussianblur_at_inference',
                        action='store_true',
                        help='If true gaussian blur is performed frame by frame at test time.')
    parser.add_argument('--gaussianblur_kernel_size',
                        default=(1, 1),
                        type=int,
                        nargs='+',
                        help='Size of the gaussian kernel.')
    parser.add_argument('--gaussianblur_sigma',
                        default=(1, 1),
                        type=float,
                        nargs='+',
                        help='Sigma of the gaussian kernel.')
    parser.add_argument('--train_t_crop',
                        default='random',
                        type=str,
                        help=('Temporal cropping method in training. '
                              'random is uniform. '
                              '(random | center | even_crops)'))
    parser.add_argument(
        '--segment_duration',
        default=8,
        type=int,
        help='Number of frames per segment. Used for even crops sampling and '
             'the temporal augmentations.')
    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening',
                        default=0.0,
                        type=float,
                        help='dampening of SGD')
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')
    parser.add_argument('--dropout',
                        default=0.0,
                        type=float,
                        help='Dropout probability')
    parser.add_argument('--mean_dataset',
                        default='kinetics',
                        type=str,
                        help=('dataset for mean values of mean subtraction'
                              '(activitynet | kinetics | 0.5)'))
    parser.add_argument('--no_mean_norm',
                        action='store_true',
                        help='If true, inputs are not normalized by mean.')
    parser.add_argument(
        '--no_std_norm',
        action='store_true',
        help='If true, inputs are not normalized by standard deviation.')
    parser.add_argument(
        '--value_scale',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Currently only support SGD')
    parser.add_argument('--lr_scheduler',
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument(
        '--multistep_milestones',
        default=[50, 100, 150],
        type=int,
        nargs='+',
        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument(
        '--overwrite_milestones',
        action='store_true',
        help='If true, overwriting multistep_milestones when resuming training.'
    )
    parser.add_argument(
        '--plateau_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Batch Size')
    parser.add_argument(
        '--inference_batch_size',
        default=0,
        type=int,
        help='Batch Size for inference. 0 means this is the same as batch_size.'
    )
    parser.add_argument(
        '--batchnorm_sync',
        action='store_true',
        help='If true, SyncBatchNorm is used instead of BatchNorm.')
    parser.add_argument('--n_epochs',
                        default=200,
                        type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--n_val_samples',
                        default=3,
                        type=int,
                        help='Number of validation samples for each activity')
    parser.add_argument('--resume_path',
                        default=None,
                        type=Path,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val',
                        action='store_true',
                        help='If true, validation is not performed.')
    parser.add_argument('--inference',
                        action='store_true',
                        help='If true, inference is performed.')
    parser.add_argument('--inference_subset',
                        default='val',
                        type=str,
                        help='Used subset in inference (train | val | test)')
    parser.add_argument('--inference_stride',
                        default=16,
                        type=int,
                        help='Stride of sliding window in inference.')
    parser.add_argument(
        '--inference_crop',
        default='center',
        type=str,
        help=('Cropping method in inference. (center | nocrop)'
              'When nocrop, fully convolutional inference is performed,'
              'and mini-batch consists of clips of one video.'))
    parser.add_argument(
        '--inference_no_average',
        action='store_true',
        help='If true, outputs for segments in a video are not averaged.')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help=
        '(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | vidbagnet |')
    parser.add_argument('--model_depth',
                        default=18,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--receptive_size',
                        default=9,
                        type=int,
                        help='Depth of resnet (1 | 9 | 17 | 33)')
    parser.add_argument('--n_input_channels',
                        default=3,
                        type=int,
                        help='Number of model input channels (default: 3).')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--wide_resnet_k',
                        default=2,
                        type=int,
                        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality',
                        default=32,
                        type=int,
                        help='ResNeXt cardinality')
    parser.add_argument('--input_type',
                        default='rgb',
                        type=str,
                        help='(rgb | flow)')
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--accimage',
                        action='store_true',
                        help='If true, accimage is used to load images.')
    parser.add_argument('--output_topk',
                        default=5,
                        type=int,
                        help='Top-k scores are saved in json file.')
    parser.add_argument('--file_type',
                        default='jpg',
                        type=str,
                        help='(jpg | hdf5 |   )')
    parser.add_argument('--tensorboard',
                        action='store_true',
                        help='If true, output tensorboard log file.')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    # temporal augmentations for CVPR 
    parser.add_argument('--augment_duration',
                        action='store_true',
                        help='If true, sampling stride is randomly sampled \
                            to alter durations.')
    parser.add_argument(
        '--augment_duration_strides',
        default=[4, 8, 16],
        type=int,
        nargs='+',
        help='Values for the sampling stride that can be sampled in augment_duration.')
    parser.add_argument('--augment_order',
                        action='store_true',
                        help='If true, video segments are shuffled.')
    parser.add_argument(
        '--augment_relative_segment_length',
        default=0.1,
        type=float,
        help='Segment length in %. Default: 10%.')
    parser.add_argument('--augment_composition',
                        action='store_true',
                        help='If true, video segments are removed to alter \
                            frequent (co-)occurrences.')
    parser.add_argument(
        '--augment_composition_prob',
        default=0.1,
        type=float,
        help='Amount of frames to remove from a video. Default: 10%.')
    # Image models for feature extraction, ICCV
    parser.add_argument('--use_image_features',
                        action='store_true',
                        help='If true, use an image model from timm to extract image features for each frame.')
    parser.add_argument('--timm_model',
                        default='xception41',
                        type=str,
                        help='If use_image_features, specifies which model from timm to use.')
    
    
    if arguments_string != None: 
        args = parser.parse_args(shlex.split(arguments_string))
    else:    
        args = parser.parse_args()

    return args
