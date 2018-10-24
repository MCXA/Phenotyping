"""
Copy & Pasted from William's Plot Vision API. Plenty of work to do yet.
"""

import os

import numpy as np
import tqdm
import argparse
import deepplantphenomics.tools
import glob
import skimage.io
import imghdr


def file_is_image(file_path):
    return imghdr.what(file_path) is not None


def scale_image(img, old_min, old_max, new_min, new_max, rint=False, np_type=None):
    """
    :param img: numpy array
    :param old_min: scalar old minimum pixel value
    :param old_max: scalar old maximum pixel value.
    :param new_min: scalar new minimum pixel value
    :param new_max: scalar new maximum pixel value.
    :param rint: Should the resulting image be rounded to the nearest int values? Does not convert dtype.
    :param np_type: Optional new np datatype for the array, e.g. np.uint16. If none, keep current type.
    :return: scaled copy of img, and scaling parameters (a, b), such that img_scaled = a * img + b.

    equivalent to:
    img = (new_max - new_min) * (img - old_min) / (old_max - old_min) + new_min
    see https://stats.stackexchange.com/a/70808/71483 and its comments.
    """
    a = (new_max - new_min) / (old_max - old_min)
    b = new_min - a * old_min
    # This autoconverts to float64, preventing over-/under-flow in most cases.
    img = a * img + b
    if rint:
        img = np.rint(img)
    if np_type:
        img = img.astype(np_type)
    return img, (a, b)


def scale_image_auto(img, new_min, new_max, rint=False, np_type=None):
    """
    :param img: numpy array
    :param new_min: scalar new minimum pixel value
    :param new_max: scalar new maximum pixel value.
    :param rint: Should the resulting image be rounded to the nearest int values? Does not convert dtype.
    :param np_type: Optional new np datatype for the array, e.g. np.uint16. If none, keep current type.
    :return: copy of img, with all pixels scaled according to the global max and min pixels (across all channels),
            tuple of (old_min, old_max).
    """
    if np.issubdtype(img.dtype, np.bool_):
        # scale_image() can't handle the bool type.
        img = np.uint8(img)
    # if this is an intermediate image, there could be nans.
    old_max = np.nanmax(img)
    old_min = np.nanmin(img)
    return scale_image(img, old_min, old_max, new_min, new_max, rint=rint, np_type=np_type)


def split_img_into_patches(img, grid_shape, standardize_block_shape=True):
    """
    Copied from https://stackoverflow.com/a/17385776/2643620.
    :param img: np array
    :param grid_shape: the number of blocks in each dimension. Imagine a matrix of M x N blocks.
    :param standardize_block_shape: Should all blocks have the same shape? If true, there will be regular slivers of
    the image that are not covered by the blocks. Otherwise, the blocks will likely vary by +/- 1 px, due to rounding
    (inevitable).

    :yield: tuple: (< current image block >, < its extent: minx, miny, maxx, maxy >)

    Each block has the same shape (+/- 1 px due to rounding).

    """
    xcut = np.linspace(0, img.shape[0], grid_shape[0] + 1).astype(np.int)
    ycut = np.linspace(0, img.shape[1], grid_shape[1] + 1).astype(np.int)
    xcut_sizes = xcut[1:] - xcut[:-1]
    min_xcut = int(min(xcut_sizes))
    ycut_sizes = ycut[1:] - ycut[:-1]
    min_ycut = int(min(ycut_sizes))

    yield min_ycut, min_xcut

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if standardize_block_shape:
                x_endpt = xcut[i] + min_xcut
                y_endpt = ycut[j] + min_ycut
            else:
                x_endpt = xcut[i + 1]
                y_endpt = ycut[i + 1]
            block = img[xcut[i]:x_endpt, ycut[j]:y_endpt]
            extent = xcut[i], ycut[j], x_endpt, y_endpt
            yield block, extent


def split_img_into_patches_of_size(img, block_shape):
    """

    :param img:
    :param block_shape: desired output shape of each block.

    :return: block count,  img split into a grid of blocks, each with the same shape (+/- 1 px due to rounding),
    where that shape
    is as close as possible to block_shape.
    """
    grid_height = int(round(img.shape[0] / block_shape[0]))
    grid_width = int(round(img.shape[1] / block_shape[1]))
    block_count = grid_height * grid_width
    blocks_gen = split_img_into_patches(img, (grid_height, grid_width))
    block_shape = next(blocks_gen)
    return block_count, block_shape, blocks_gen


def vegetation_segmentation_wrapper(paths, patch_shape, batch_size=8):
    return deepplantphenomics.tools.segment_vegetation(paths, batch_size=batch_size, img_height=patch_shape[0],
                                                       img_width=patch_shape[1])


def apply_in_patches_with_one_to_one_results(f, paths, patch_shape, tmp_dir_path, result_dir,  *args, **kwargs):
    out_paths = []
    for path in tqdm.tqdm(paths):
        basepath, ext = os.path.splitext(path)
        basename = os.path.basename(basepath)
        img = skimage.io.imread(path)
        if len(img.shape) == 1:
            img = img[0]
        patch_count, patch_shape, patches = split_img_into_patches_of_size(img, patch_shape)

        extents = []
        patch_paths = []
        for i, (block, extent) in zip(tqdm.trange(patch_count, desc='Creating patches...'), patches):
            out_path = os.path.join(tmp_dir_path, basename + '_' + str(i).zfill(6) + '.png')
            skimage.io.imsave(out_path, block)
            patch_paths.append(out_path)
            extents.append(extent)
        results_arr = f(patch_paths, patch_shape, *args, **kwargs)

        if results_arr is None:
            raise RuntimeError("No results from function {}".format(f.__name__))

        result_img = np.zeros(img.shape[0:2]).astype(results_arr.dtype)
        for row, (xmin, ymin, xmax, ymax) in zip(results_arr, extents):
            result_img[xmin: xmax, ymin: ymax] = row
        result_img_viewable, _ = scale_image_auto(result_img, 0, 255, True, 'uint8')
        out_path = os.path.join(result_dir, basename + '_' + f.__name__ + '.png')
        skimage.io.imsave(out_path, result_img_viewable)
        out_paths.append(out_path)
    return out_paths


def main(images_dir, patch_shape=(256, 256), patches_dir=None):
    image_paths = [path for path in glob.iglob(os.path.join(images_dir, '*'))
                   if not os.path.isdir(path) and file_is_image(path)]
    if not patches_dir:
        patches_dir = os.path.join(images_dir, 'patches')
        os.makedirs(patches_dir, exist_ok=True)
    result_dir = os.path.join(images_dir, 'results')
    apply_in_patches_with_one_to_one_results(vegetation_segmentation_wrapper, image_paths, patch_shape, patches_dir,
                                             result_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir", help="path to images dir.")
    ap.add_argument("--patch_shape", help="optional string representing a python tuple of (img_height, img_width). "
                                          "e.g. '(256, 256)'.", default="(256, 256)")
    ap.add_argument("--patches_dir", help="optional path to dir to which to store patches. "
                                          "Otherwise will be mapped to $images_dir/patches/.")

    parsed_args = ap.parse_args()

    mpatch_shape = tuple(int(x) for x in eval(parsed_args.patch_shape))

    main(parsed_args.images_dir, mpatch_shape, parsed_args.patches_dir)
