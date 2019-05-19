import argparse
import cv2
import math
import os
import random
from heapq import heappush, heappop
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

hooked_out = None


def forward_hook(module: nn.Module, tensor_in: torch.Tensor, tensor_out: torch.Tensor):
    """
    Store results of forward pass
    """
    global hooked_out
    hooked_out = tensor_out


def load_batch(jpegs: list, image_index: int, batch_size: int, resolution: int, device: torch.device) -> torch.Tensor:
    """
    Loads batch of data
    :param jpegs: list of image paths
    :param image_index: index of first image to use
    :param batch_size: count of images in the batch
    :param resolution: spatial resolution
    :param device: torch device
    :return: torch.Tensor batch
    """

    batch_list = []
    for i in range( batch_size ):
        jpeg_filename = jpegs[i + image_index]
        image = cv2.imread( jpeg_filename )
        resized = cv2.resize( image, (resolution, resolution), interpolation=cv2.INTER_AREA )
        batch_list.extend( cv2.split( resized ) )

    np_array = np.array( batch_list, dtype=np.float32 ).reshape( batch_size, 3, resized.shape[0], -1 ) / 255
    return torch.as_tensor( np_array ).to( device )


def get_jpegs_from_dir(dir_name: str, extensions: set = {'.jpg', '.png'}) -> list:
    """
    Searches for images in a directory
    :param dir_name: directory path
    :param extensions: image extensions
    :return: list of full paths
    """

    files = []
    all_files = os.listdir( dir_name )
    for name in all_files:
        if extensions is not None:
            ext = os.path.splitext( name )[1]
            if ext.lower() in extensions:
                files.append( os.path.join( dir_name, name ) )
        else:
            files.append( os.path.join( dir_name, name ) )

    return [f.replace( '\\', '/' ) for f in files]


def extract_patch_chw(image: np.ndarray, patch_center_y: int, patch_center_x: int, patch_size: int) -> np.ndarray:
    """
    Extract patch from channel-first image
    :param image: source image
    :param patch_center_y: patch center Y
    :param patch_center_x: patch center X
    :param patch_size: size of patch
    :return: extracted patch
    """

    x0 = patch_center_x - patch_size // 2
    y0 = patch_center_y - patch_size // 2
    x1 = patch_center_x + patch_size // 2 + 1
    y1 = patch_center_y + patch_size // 2 + 1

    dst_x0 = 0
    dst_y0 = 0
    dst_x1 = patch_size
    dst_y1 = patch_size

    # clipping
    if x0 < 0:
        dst_x0 = -x0
        x0 = 0

    if y0 < 0:
        dst_y0 = -y0
        y0 = 0

    if len( image.shape ) == 3:
        # 3-channel image
        image_width = image.shape[2]
        image_height = image.shape[1]
        num_channels = image.shape[0]
        patch_shape = (num_channels, patch_size, patch_size)
    elif len( image.shape ) == 2:
        # single channel image
        image_width = image.shape[1]
        image_height = image.shape[0]
        num_channels = 1
        patch_shape = (patch_size, patch_size)
    else:
        raise Exception( 'Unsupported image.shape :' + str( image.shape ) )

    if x1 > image_width:
        dst_x1 -= x1 - image_width
        x1 = image_width

    if y1 > image_height:
        dst_y1 -= y1 - image_height
        y1 = image_height

    patch = np.zeros( patch_shape, image.dtype )

    if num_channels != 1:
        patch[:, dst_y0: dst_y1, dst_x0: dst_x1] = image[:, y0: y1, x0: x1]
    else:
        patch[dst_y0: dst_y1, dst_x0: dst_x1] = image[y0: y1, x0: x1]

    return patch


def convert_to_8bpp(arr_float: np.ndarray) -> np.ndarray:
    """
    Converts float image to uint8 image
    :param arr_float: float image
    :return: uint8 image
    """
    return np.clip( arr_float, 0, 255 ).astype( np.uint8 )


def show_all_activations(best_patches: dict, num_neurons: int, topn_sqrt: int, patch_size: int):
    image_width = 2560
    image_height = 4560

    pad = 10
    inner_space = 4

    first_patch = best_patches[0][0][1]

    image = np.ndarray( (image_height, image_width, first_patch.shape[2]), dtype=np.uint8 )
    image.fill( 255 )

    grid_nx = int( math.sqrt( num_neurons ) * image_width / image_height )
    grid_ny = math.ceil( num_neurons / grid_nx )

    cell_width = (image_width - 2 * pad) / grid_nx
    cell_height = (image_height - 2 * pad) / grid_ny

    _cell_inner_width = int( cell_width - inner_space )
    _cell_inner_height = int( cell_height - inner_space )

    # making inner cell square
    cell_inner_size = min( _cell_inner_width, _cell_inner_height )

    for j in range( grid_ny ):
        cell_y0 = int( pad + cell_height * j )
        for i in range( grid_nx ):
            cell_x0 = int( pad + cell_width * i )

            neuron_index = i + j * grid_nx
            if neuron_index >= num_neurons:
                break

            kernel_image = get_neuron_image( best_patches, neuron_index, topn_sqrt, patch_size )
            interpolation = cv2.INTER_AREA if cell_inner_size < kernel_image.shape[0] else cv2.INTER_LINEAR
            img_resized = cv2.resize( kernel_image, (cell_inner_size, cell_inner_size), interpolation=interpolation )
            image[cell_y0: cell_y0 + cell_inner_size, cell_x0: cell_x0 + cell_inner_size] = img_resized

    return image


def get_neuron_image(all_patches: dict, neuron_index: int, nx: int, patch_size: int) -> np.ndarray:
    ny = nx

    patch_width = patch_size
    patch_height = patch_size

    img_width = nx * patch_width
    img_height = ny * patch_height

    img = np.ndarray( (img_height, img_width, 4), np.uint8 )

    if neuron_index not in all_patches:
        return img

    patches = all_patches[neuron_index]

    for i in range( len( patches ) ):
        y = (i // nx) * patch_height
        x = (i % nx) * patch_width

        img[y: y + patch_height, x: x + patch_width] = patches[i][1]

    return img


def is_in_top_n(neurons: dict, neuron_index: int, max_activation: float, top_n: int):
    if max_activation <= 0:
        return False

    if neuron_index not in neurons:
        return True

    stored_maximums = neurons[neuron_index]
    if len( stored_maximums ) < top_n:
        return True

    min_activation = stored_maximums[-1][0]
    return max_activation > min_activation


def update_top_n(neurons: dict, neuron_index: int, max_activation: float, patch, top_n: int):
    if max_activation <= 0:
        return

    if neuron_index not in neurons:
        neurons[neuron_index] = [(max_activation, patch)]
        return

    stored_maximums = neurons[neuron_index]
    heappush( stored_maximums, (max_activation, patch) )

    if len( stored_maximums ) > top_n:
        heappop( stored_maximums )


def show_max_activations(args):
    use_cuda = torch.cuda.is_available() and args
    device = torch.device( "cuda" if use_cuda else "cpu" )
    print("Using GPU" if use_cuda else "Using CPU")

    use_grads = args.grads

    # searching for images
    jpegs = get_jpegs_from_dir( args.images_dir )
    random.shuffle( jpegs )
    images_count = args.images_count
    if images_count < 0:
        images_count = len( jpegs )
    else:
        images_count = min( images_count, len( jpegs ) )

    print('Using %d images' % images_count)

    # print('Loading model...')

    model = args.model

    # model_name = args.model
    # model = pretrainedmodels.__dict__[model_name]( num_classes=1000, pretrained='imagenet' ).to( device )

    # searching for layer
    submodules = args.layer.split( '.' )
    module = model
    for submodule in submodules:
        module = getattr( module, submodule, None )
        if module is None:
            print('Layer "%s" not found in model' % args.layer)

    # registering forward hook
    module.register_forward_hook( forward_hook )

    topn = args.topn_sqrt ** 2
    best_patches = dict()

    num_neurons_printed = False

    # running batches
    for image_index in range( images_count ):

        print('%1.1f%%, running %s' % (100. * image_index / images_count, jpegs[image_index]))

        # loading batch
        image_tensor = load_batch( jpegs, image_index, 1, args.resolution, device )

        image_height = image_tensor.shape[2]
        image_width = image_tensor.shape[3]

        # performing forward pass
        image_tensor.requires_grad_( True )
        model.forward( image_tensor )

        # searching max activation
        num_neurons = hooked_out.shape[1]
        act_np = hooked_out[0].detach().cpu().numpy()
        flat_activations = act_np.reshape( act_np.shape[0], -1 )
        max_indexes = np.argmax( flat_activations, 1 )

        if not num_neurons_printed:
            print('Number of neurons = %d' % num_neurons)
            print('Feature map is %d x %d' % (hooked_out.shape[2], hooked_out.shape[3]))
            num_neurons_printed = True

        feature_map_height = hooked_out.shape[2]
        feature_map_width = hooked_out.shape[3]

        # extracting patches
        for neuron_index in range( num_neurons ):
            max_index = max_indexes[neuron_index]
            max_act = flat_activations[neuron_index, max_index]

            if not is_in_top_n( best_patches, neuron_index, max_act, topn ):
                continue

            center_y = max_index // feature_map_width
            center_x = max_index % feature_map_width

            patch_center_y = (center_y * image_height) // feature_map_height
            patch_center_x = (center_x * image_width) // feature_map_width

            if use_grads:
                # doing backward passes to get gradient
                hooked_out[0, neuron_index, center_y, center_x].backward( retain_graph=True )

                # getting absolute gradient
                grads = image_tensor.grad.data.clone().detach()
                grads *= grads

                # RGB grads
                grads = grads.reshape( (3, image_height, image_width) )

                # gray grads
                grads_gray = torch.sqrt( grads.sum( dim=0 ) )
                grads_gray = torch.sqrt( grads_gray )

                # zeroing gradients
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

            else:
                grads_gray = torch.ones( (1, image_height, image_width) )

            _rgb = (image_tensor[0].clone().detach()).cpu().numpy()
            _grags_gray_np = grads_gray.detach().cpu().numpy().reshape( (image_height, image_width) )

            # extracting patches
            _rgb = extract_patch_chw( _rgb, patch_center_y, patch_center_x, args.patch_size )
            grags_gray_np = extract_patch_chw( _grags_gray_np, patch_center_y, patch_center_x, args.patch_size )

            # adding gradients as alpha channel
            grags_gray_np = cv2.blur( grags_gray_np, (7, 7) )
            cv2.normalize( grags_gray_np, grags_gray_np, 1, 0, cv2.NORM_MINMAX )
            rgba = np.append( _rgb, grags_gray_np.reshape( (1, args.patch_size, args.patch_size) ), axis=0 )
            blended = convert_to_8bpp( cv2.merge( rgba * 255 ) )

            update_top_n( best_patches, neuron_index, max_act, blended, topn )

    image = show_all_activations( best_patches, num_neurons, args.topn_sqrt, args.patch_size )
    cv2.imwrite( './data/activations' + '_' + args.layer + '.png', image )
    return image


def get_layers_weight_counts(model: nn.Module) -> OrderedDict:
    """
    Returns map: layer name -> weights count. Map is sorted on ascending count of weights
    :return: OrderedDict of layer name -> weights count
    """

    names = []
    counts = []
    for name, value in model.named_parameters():
        if 'weight' in name:
            count = 1
            for i in range( len( value.shape ) ):
                count *= value.shape[i]
            # count = value.numel()
            names.append( name )
            counts.append( count )

    result = OrderedDict()
    indexes = np.argsort( counts )
    for i in indexes:
        name = names[i]
        count = counts[i]
        result[name] = count

    return result


def get_layers_weight(model: nn.Module) -> dict:
    """
    Returns map: layer name -> weights.
    :return: Dict of layer name -> weights
    """

    names = []
    weights = []
    for name, value in model.named_parameters():
        if 'weight' in name:
            weights.append( value.flatten().detach().cpu().numpy() )
            names.append( name )

    result = dict( zip( names, weights ) )

    return result


def print_summary(model: nn.Module):
    #print('Loading model...')
    #model = pretrainedmodels.__dict__[model_name]( num_classes=1000, pretrained='imagenet' )

    weight_counts = get_layers_weight_counts( model )
    total_weights = 0
    for name in weight_counts:
        count = weight_counts[name]
        total_weights += count
        print("%s: %d" % (name, count))

    print('\nTotal weights count = %d' % total_weights)


# print_summary( 'resnet18' )


class Args:
    def __init__(self, model: nn.Module, layer: str, directory_name: str,  patch_size: int = 31):
        self.grads = True
        self.model = model
        self.resolution = 224
        self.images_dir = directory_name
        self.layer = layer
        self.gpu = True
        self.images_count = -1
        self.topn_sqrt = 3
        self.patch_size = patch_size


def show_max_activations_patches(model: nn.Module, layer: str, directory_name: str) -> None:
    image = show_max_activations(Args(model, layer, directory_name))
    plt.figure(figsize=(30, 60))
    plt.imshow(image)
    plt.show()


def show_distributions_of_layers(model: nn.Module) -> None:
    weigts = get_layers_weight(model)

    for name in weigts.keys():
        plt.hist(weigts[name], bins=30, color='g')
        plt.xlabel('weight')
        plt.title(name)
        plt.show()


# for i in range( 1, 3 ):
#     show_max_activations( Args( 'resnet18', 'layer' + str( i ) ) )

# model_name = 'resnet18'
# model = pretrainedmodels.__dict__[model_name]( num_classes=1000, pretrained='imagenet' )
# weigts = get_layers_weight( model )
#
# len( weigts.keys() )
#
#
# for name in weigts.keys():
#     plt.hist( weigts[name], bins=30, color='g' )
#     plt.xlabel( 'weight' )
#     plt.title( name )
#     plt.show()
