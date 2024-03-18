#!/usr/bin/env python

"""
This package contains a list of functions that can be used to load and reduce images. This includes images taken for astrophotography, grating spectroscopy, and stellar photometry.
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.visualization import simple_norm

import warnings
from PIL import Image
import rawpy

from matplotlib.patches import Rectangle
import glob as glob
from os import path as ospath, symlink as ossymlink, unlink as osunlink
from pathlib import Path


def load_image(filename, gray_conv = [0.2126, 0.7152, 0.0722], bayer_colors=["R", "G", "B"], return_filtered_raw=True, print_log=False, cmap='Greys_r', stretch='log', log_a=1000):
    """
    Load an image by filename and produce a 2D array.

    Input images can either be FITS, RAW (NEF), or JPG images. If JPG images are provided, they are converted to grayscale first through a grayscale conversion vector that can be modified. RAW images are read using the ``rawpy`` package, and data for each of the individual colour filters is returned.

    .. note:: The original implementation of reading RAW (NEF) image data was due to Ayush Gurule, an Ashoka undergraduate from the UG25 (2022-2026) batch.

    Parameters
    ----------
    filename: str
        The image file to read: a filename pointing to the location of the file.

    gray_conv: [float, float, float], default: [0.2126, 0.7152, 0.0722] 
        R, G, and B weights to convert a JPEG or PNG from RGB to grayscale.

    bayer_colors: list of chars, default: ["R", "G", "B"]
        Colours to return, as defined in the Bayer pattern of a RAW image.

    return_filtered_raw: bool, default: True
        When extracting pixel-level information for individual colours in a Bayered sensor, the resulting arrow will contain multiple zeros for the rows and columns of pixels of a different colour from the one requested. By default, all these zeros are removed and the array is returned.

        .. note:: Removing these pixels effectively "reduces" the size of the image by half in each dimension.

    print_log: bool, default: False
        Provides the option to print a log to debug your code. In this case, whether you want display the image you have loaded. For JPG or RAW images with multiple channels, a monochrome "average" over the channels is plotted.

    cmap: str, default: "Greys_r"
        A string containing the colormap title. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.

    stretch: "log" or None, default: "log"
        A string describing the type of stretching that can be applied to display image data. By default, a log stretch is applied. # TODO: Add more stretches.

    log_a: float, default: 1000
        The log index for ``stretch='log'``.

    Returns
    -------
    data: array_like
        An array containing the image data. If the input image is a RAW file, then this array will have one 2D array for each element of ``bayer_colors``, which can be unpacked into as many variables. For all other input images, a single 2D array is returned.

    Raises
    ------
    ValueError
        If the filetype is not FIT, FITS, JPG, JPEG, PNG, or NEF.

    ValueError
        If the input image is a RAW (NEF) file, and any element of ``bayer_colors`` is not present in the image's Bayer pattern.

    Usage
    -----
    >>> this_data = load_image("some_image.JPG")
    """
    extension = filename.split(".")[-1].upper()             # Get the file extension from the string

    if(print_log):                                          # Print detected extension in the log
        print("File extension detected as", extension)

    if(extension=="FIT" or extension=="FITS"):              # If the file is a fits file
        data = fits.getdata(filename)                       # use astropy to load the fits data
    elif(extension=="JPG" or extension=="JPEG" or extension=="PNG"):
        data_rgb = plt.imread(filename)                     # For jpg and pngs, use ``plt.imread``.
        data = gray_conv[0]*data_rgb[:,:,0] + \
               gray_conv[1]*data_rgb[:,:,1] + \
               gray_conv[2]*data_rgb[:,:,2]                 # Weight and add the R,G,B channels
    elif(extension=="NEF"):
        rawpy_object = rawpy.imread(filename) # Load the RAW image as a RawPy object
        raw_image = rawpy_object.raw_image    # Extract the image data from the RawPy object

        bayer = np.array(list(rawpy_object.color_desc.decode('ascii'))) # Get the Bayer pattern for this camera
        bayer_index = np.arange(len(bayer))   # Define the Bayer index for each color

        raw_colors = rawpy_object.raw_colors  # An array in which each element (pixel) is marked by its Bayer index For example: all ``0``s are "Red" and all ``1``s are "Green" (say).

        skip_rows, skip_cols = np.shape(rawpy_object.raw_pattern) # Size of the smallest repeatable Bayer pattern of the image

        if(not np.isin(bayer_colors, bayer).all()): # Throw an error if the requested colours and Bayer pattern aren't the same.
            raise ValueError(f"`bayer_colors` {bayer_colors} must contain elements from the Bayer pattern of the image ({bayer}).")

        bayer_data = []  # Array to hold the image data for each Bayer colour

        for i in bayer_index:                           # For each Bayer index,
            mask=np.array((raw_colors==bayer_index[i])) # create a mask,
            tmp_image = mask * raw_image                # extract data from mask
            bayer_data.append(tmp_image)                # and store the output in ``bayer_array``.

            # Note: The ``tmp_image`` above will have multiple zeros for the rows and columns of pixels of a different colour from the one requested. If ``return_filtered_raw=True``, only the non-zero pixels are returned.

            if(return_filtered_raw):
                # Find the row and column index of the requested colour from the Bayer pattern array.
                row_index, col_index = np.argwhere(rawpy_object.raw_pattern==i)[0]
                # Take only one pixel from each Bayer pattern by skipping rows and columns.
                bayer_data[i] = bayer_data[i][row_index::skip_rows,col_index::skip_cols]

        bayer_data = np.array(bayer_data) # Convert to a NumPy array to simplify operations later.

        data = [] # Array to store values finally returned.

        for color in bayer_colors: # For each requested colour
            c_index = np.nonzero(bayer==color)[0] # Find index in ``bayer_data``
            # Note: Some Bayer patterns have multiple pixels of the same colour (like Green, for example). If so, multiple indices are returned.
            data.append(np.mean(bayer_data[c_index], axis=0)) # If multiple elements exist, average over them.
        data = np.array(data)
    else:
        raise ValueError("Filetype not supported: "+extension)

    if(print_log):                                          # Display image, if a log is requested
        display_data = np.sum(data, axis=0) if extension == "NEF" else data # For NEF data, divide by the max to get values in (0,1)
        display(display_data, cmap=cmap, stretch=stretch, log_a=log_a)

    return data.astype(np.float32)


def display(image_array, cmap='Greys_r', stretch='log', log_a=1000, norm_array=None, min_percent=0.0, max_percent=100.0, title=None, xlim=None, ylim=None, fig=None, ax=None):
    """
    Display 2D scalar data as an image.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    cmap: str, default: "Greys_r"
        A string containing the colormap title. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.

    stretch: {"linear", "sqrt", "power", "log", "asinh"}, default: "log"
        A string describing the type of stretching that can be applied to display image data. By default, a log stretch is applied.

    log_a: float, default: 1000
        The log index for ``stretch='log'``.

    norm_array: array_like, default: None
        A 2D array used to decide the normalisation of the ``simple_norm`` (from ``astropy.visualization``) used to visualise the plot. By default, the ``image_array`` is used for this norm.

    min_percent : float, default: 0.0
        The percentile value used to determine the pixel value of minimum cut level. This is a parameter for the ``astropy.visualization.simple_norm`` used to display the image.

    max_percent : float, default: 100.0
        The percentile value used to determine the pixel value of maximum cut level.  This is a parameter for the ``astropy.visualization.simple_norm`` used to display the image.

    xlim, ylim: float, default: None
        Set the ``x`` or ``y`` limits of the plot. First element is the lower limit, and the second is the upper limit.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    matplotlib.image.AxesImage (A plot).

    Raises
    ------
    ValueError
        If the ``stretch`` provided is not one of the allowed stretches.

    Usage
    -----
    >>> display(image_data, cmap="inferno", stretch='log', log_a=10, xlim=None, ylim=None, fig=None, ax=None)
    """
    allowed_stretches = ["linear", "sqrt", "power", "log", "asinh"]

    if(stretch not in allowed_stretches):
        raise ValueError("The chosen stretch \""+ str(stretch) +"\" is not implemented. The allowed values of stretches are: "+str(allowed_stretches))
    
    if(fig is None or ax is None):
        fig, ax = plt.subplots()            # TODO: Incorporate figsize somehow?

    if(norm_array is None):                 # If no explicit ``norm_array`` is provided, use
        norm_array = image_array            # the ``image_array`` to compute the display norm
        
    img = ax.imshow(image_array, origin='lower', cmap = cmap, norm=simple_norm(norm_array, stretch=stretch, log_a=log_a, min_percent=min_percent, max_percent=max_percent))               # Display the image

    ax.set_xlim(xlim)                                     # Set the ``xlim`` of the plot
    ax.set_ylim(ylim)                                     # Set the ``ylim`` of the plot

    if(title is not None):
        ax.set_title(title)

    return img


def get_files(pathname, root_dir=None, print_log=False):
    """
    Get all filenames that match a given pattern and return a sorted list.

    This is a simple wrapper around the ``glob.glob`` function. From ``glob.glob``'s docstring: Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la fnmatch. However, unlike fnmatch, filenames starting with a dot are special cases that are not matched by '*' and '?' patterns.

    If recursive is true, the pattern '**' will match any files and zero or more directories and subdirectories.

    Parameters
    ----------
    pathname: str
        A string containing a path specification. Can be either absolute (like ``~/data/ring*L.txt``) or relative (like ``../../spectrum_sirius*.fit``), and can contain shell-style wildcards (* and ?). 

    root_dir: str or None, default: None
        Path specifying the root directory for searching.  If ``pathname`` is relative, the result will contain paths relative to ``root_dir``.

    print_log: bool, default: False
        Provides the option to print a log to debug your code. In this case, it will print out the list of files loaded.

    Returns
    -------
    files: array_like
        A 1D array containing the paths of all the files that match a specific pattern.

    Usage
    -----
    >>> filelist = get_files("./data/imaging/ring*L.fit")
    """
    files = np.sort(glob.glob(pathname, root_dir=root_dir))

    if(print_log):
        print("Files loaded:\n", files)

    return files


def stack_files(filelist, stack_type='mean', gray_conv=[0.2126, 0.7152, 0.0722], bayer_colors=["R", "G", "B"], return_filtered_raw=True, print_log=False):
    """
    Stack multiple image files together to produce a single output file. Input files can either be a list of FITS, NEF, or JPG images. If JPG images are used, they are converted to grayscale first using the ``gray_conv`` parameter. If a list of RAW images is provided, the stacked results for each of the individual colour filters is returned.

    Parameters
    ----------
    filelist: array_like
        List of strings containing the filenames of files that need to be added.

    stack_type: {"mean", "average", "sum"}, default: "mean"
        A string describing the type of stacking. "average" and "mean" both perform the same action.

    bayer_colors: list of chars, default: ["R", "G", "B"]
        Colours to return, as defined in the Bayer pattern of a RAW image.

    return_filtered_raw: bool, default: True
        Return only pixel-values of colours specified in ``bayer_colors`` from a RAW image, ignoring all other pixels.

    print_log: bool, default: False
        A boolean variable to decide whether you want to print a log of what you've done. In this case, a list of the files stacked.

    Returns
    -------
    stacked: array_like
        An array containing the sum (or average) of every item in the file-list.  If the input images are RAW files, this array will have one 2D array for each element of ``bayer_colors``, which can be unpacked into as many variables. For all other input images, a single 2D array is returned.

    Raises
    ------
    ValueError
        If ``stack_type`` is not one of the available types.

    Usage
    -----
    >>> stacked_image = stack_files(filelist, stack_type='mean')
    """
    stacked = np.zeros_like(load_image(filelist[0], return_filtered_raw=return_filtered_raw), dtype=np.float32) # Create an empty array of zeros
    
    for i in range(len(filelist)):                              # Load each file,
        stacked += load_image(filelist[i], gray_conv=gray_conv, print_log=print_log, bayer_colors=bayer_colors, return_filtered_raw=return_filtered_raw) # and add it to ``stacked``
        if(print_log):
            print("File completed:", filelist[i])

    if(stack_type=='mean' or stack_type=='average'):
        stacked = stacked/len(filelist) # Divide by number of files to obtain the mean
    elif(stack_type != 'sum'):
        raise ValueError("`stack_type` should be either \"sum\" or \"mean\" (or \"average\").")

    return stacked


def stack(array_of_images, shifts=None, stack_type='mean', print_log=False):
    """
    Stack an array of multiple image arrays together to produce a single output array. Each image may be shifted by a pre-specified amount before stacking. Images shifts are taken from the ``shifts`` variable.
    
    **WARNING:** Shifting an image may cause parts of the image array to be cut off.

    Parameters
    ----------
    array_of_images: array_like 
        A 3-dimensional ``(N, Ly, Lx)`` array containing ``N`` image arrays of size ``(Ly, Lx)`` to stack. 

    shifts: array_like or None, default: None
        An ``(N, 2)`` array containing ``N`` 2D vectors representing the shifts by which each image should be displaced before summing. If ``None`` is provided, no shifting is done.

    stack_type: {"mean", "average", "sum"}, default: "mean"
        A string describing the type of stacking. "average" and "mean" both perform the same action.

    print_log: bool, default: False
        A boolean variable to decide whether you want to print a log of what you've done.

    Returns
    -------
    stacked_array: array_like
        A 2D array containing the sum (or average) of every item in the file-list.

    Raises
    ------
    ValueError
        If ``stack_type`` is not one of the available types.

    Warns
    -----
    UserWarning
        If the ``array_of_images`` is not of the specified dimensions.
        
    UserWarning
        If the lengths of the ``array_of_images`` and ``shifts`` don't match.

    Usage
    -----
    >>> stacked_array = stack(filelist, stack_type='mean')
    """
    shape = np.shape(array_of_images) # Get the shape of the input array.
    
    if(len(shape)!= 3):
        warnings.warn("`array_of_images` is expected to be an array of (N, Ly, Lx) elements. Returning NoneType.", UserWarning)
        return None

    N = shape[0] # N is the first element of the `shape` array.

    ###### Two cases: ######
    #
    # CASE 1: If `shifts` aren't provided, just add up the individual arrays using `np.sum`. The normalisation (mean,
    # sum, etc.) will be done later.
    
    if(shifts is None):
        stacked_array = np.sum(array_of_images, axis=0) # If no shifts are provided, simply add up all `N` elements.
        if(print_log):
            print("No shifts provided, adding all images without shifting.")

    # CASE 2: If `shifts` are provided, first check that the right number of shifts are given.
    # If the wrong number is given, replace `shifts` by an array of zeros. Then, shift each array by the appropriate 
    # amount and add them.
    
    else:
        if(len(shifts) != N):
            warnings.warn("The lengths of `array_of_images` and `shifts` must be the same. Not applying shifts.", UserWarning)
            shifts = np.zeros((N,2)) # If lengths don't match, replace shifts with an array of zeros.

        stacked_array = np.zeros_like(array_of_images[0], dtype=np.float64) # Create a dummy array of zeros
    
        for i in range(N): # For each image
            shifted_array = shift(array_of_images[i], shifts[i]) # Shift by the appropriate amount
            stacked_array += shifted_array # Add them to `stacked_array`
    
            if(print_log):
                print("Shifted image",i+1, "by vector", shifts[i],".")
    ###### End cases ######
    
    if(stack_type=='mean' or stack_type=='average'):
        stacked_array = stacked_array/N
    elif(stack_type != 'sum'):
        raise ValueError("`stack_type` should be either \"sum\" or \"mean\" (or \"average\").")

    return stacked_array
    

def flip(image_array, axis="x"):
    """
    Flip an image along the "x" or "y" axes.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    axis: str, default: "x"
        Axis about which to flip image.
    
    Returns
    -------
    flipped_array: array_like
        A 2D array. The flipped array.

    Raises
    ------
    ValueError
        If ``axis`` is neither "x" nor "y".

    Usage
    -----
    >>> flipped_array = flip(this_data)
    """
    if(axis=="x"):
        return image_array[:,::-1]
    elif(axis=="y"):
        return image_array[::-1,:]
    else:
        raise ValueError("Can't flip around \""+axis+"\" axis." )
    
        
def crop(image_array, left=None, right=None, top=None, bottom=None, origin=None, print_log=False, fig=None, ax=None):
    """
    Crop an image between pre-specified pixels, with the option of choosing an origin around which to perform the cropping.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    left, right, top, bottom: int
        Left, right, top, and bottom pixel values to crop the image if ``origin=None``. If ``origin`` is provided, these are the pixels to the left, right, top, and bottom of the ``origin`` pixel within which the image is to be cropped.

    origin: List [int, int] or None, default: None
        Point about which to crop the image. By default, no point is provided, and ``left``, ``right``, ``top``, and ``bottom`` correspond to absolute pixel values.

    print_log: bool, default: False
        Provides the option to print a log to debug your code. In this case, it will display the cropped array.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    cropped_array: array_like
        A 2D array. The cropped array.

    Usage
    -----
    >>> cropped_array = crop(this_data, left=0, right=1500)
    >>> cropped_array = crop(this_data, left=100, right=100, top=100, bottom=100, origin=[1052,1002])
    """
    if(origin is None): # If no origin is given,
        cropped_array = image_array[bottom:top, left:right] # crop image between endpoints
    else: # If an origin is provided, use the pixels as distances from this origin
        # If any of the offsets are None, reset them to the image edges
        bottom = origin[1] if bottom is None else bottom
        top    = len(image_array) - origin[1] if top is None else top
        left   = origin[0] if left is None else left
        right  = len(image_array[0]) - origin[0] if right is None else right

        cropped_array = image_array[origin[1]-bottom:origin[1]+top, origin[0]-left: origin[0]+right]

    if(print_log):
        display(cropped_array, fig=fig, ax=ax)

    return cropped_array


def shift(image_array, displacement, print_log=False):
    """
    Translate an image by a ``displacement`` vector.
    
    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    displacement: List [float, float] 
        Vector of the (x,y) shift by which to translate the image.

    print_log: bool, default: False
        Provides the option to print a log to debug your code. In this case, it will display the shifted array.
    
    Returns
    -------
    shifted_array: array_like
        A 2D array. The rotated array.

    Usage
    -----
    >>> shifted_array = shift(this_data, displacement=[10,200])
    """
    a = 1
    b = 0
    c = displacement[0] #left/right displacement (i.e. 5/-5)
    d = 0
    e = 1
    f = displacement[1] #up/down displacement (i.e. 5/-5)
    
    # This function uses the ``Image.AFFINE`` transformation. In addition to simply translating the image, it can also "shear" it. However, we will just use it to translate the image by a fixed amount, which is why many of the variables above are 0 or 1.

    shifted_image = Image.fromarray(image_array)
    shifted_array = shifted_image.transform(shifted_image.size, Image.AFFINE, (a, b, c, d, e, f))
    shifted_array = np.array(shifted_array)

    if(print_log):
        display(shifted_array, norm_array=image_array)

    return shifted_array
    

def rotate(image_array, angle, origin=None, expand=False, fill=False, print_log=False):
    """
    Rotate an image by a fixed angle, with the option of choosing an origin about which to perform the rotation.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    angle: float
        Angle by which to rotate the image.
    
    origin: List [int, int] or None, default: None
        Point about which to rotate the image. By default, no point is provided, and the image is rotated about its centre.

    expand: bool, default: False
        Expands the output image to make it large enough to hold the entire rotated image.
    
    fill: bool, default: False
        Fill area outside the rotated image. If ``True``, this area is filled with the median value of the image.

    print_log: bool, default: False
        Provides the option to print a log to debug your code. In this case, it will display the rotated array.
    
    Returns
    -------
    rotated_array: array_like
        A 2D array. The rotated array.

    Usage
    -----
    >>> rotated_array = rotate(this_data, angle=23.0)
    >>> rotated_array = rotate(this_data, angle=23.0, origin=[1052,1002])
    """
    peak = np.max(image_array)
    image   = Image.fromarray(image_array/peak) # TODO: This is done to avoid type-casting issues. Needs to be fixed.
    fillcolor= np.median(image_array/peak) if fill else None # Fill expanded region with the median value
    r_image = image.rotate(angle, resample=Image.BICUBIC, center=origin, expand=expand, fillcolor=fillcolor) # Rotate image about star position
    rotated_array = np.asarray(peak*r_image)

    if(print_log):
        display(rotated_array, norm_array=image_array)
    
    return rotated_array


def find_star(image_array, star_pos=None, search=500, print_log=False, fig=None, ax=None):
    """
    Find the pixel location of star in a 2D image array. A "star" is defined simply as the "brightest" pixel in the array. 
    
    If multiple bright pixels can be found, a rough location and search area can be defined to refine the search.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    star_pos: [int, int] or None, default: None
        ``x`` and ``y`` pixel coordinates of the star's rough location, to refine search for the brightest pixel. If ``None`` is provided, the brightest pixel is used.

    search: float, default: 500
        Search "radius" in pixels. The brightest pixel will be found in the range (star_pos[0] +/- search, star_pos[1]+/- search).

    print_log: bool, default: False
        Provides the option to print a log to debug your code. In this case, it will show the location of the detected star, and the provided search-box.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    star: array_like
        An array of 2 elements containing ``x`` and ``y`` pixel location of the "star".

    Usage
    -----
    >>> this_star = find_star(this_data, star_pos=[1000,1500], search=500)
    """
    if(star_pos is not None): # If both ``x`` and ``y`` coordinates are provided
        # Create a trimmed array from (star_pos[0] +/- search, star_pos[1]+/- search).
        # Note: arrays are indexed first by their "y" coordinate, and then the "x" coordinate.
        trimmed_array = image_array[ star_pos[1] - search//2 : star_pos[1] + search//2, 
                                     star_pos[0] - search//2 : star_pos[0] + search//2 ] 
        is_trimmed=True  # Boolean to record if trimming has occurred
    else:
        trimmed_array = image_array # If no star position is given, search entire image.
        is_trimmed=False
    
    star_y, star_x = np.where(trimmed_array == np.max(trimmed_array)) # Find location of maximum value in `trimmed_array`
    # Note: If the array has been trimmed, these coordinates need to be transformed to the original image's coordinates
    #       this is done below.
    star = [star_x[0] + star_pos[0] - search//2, star_y[0] + star_pos[1] - search//2] if is_trimmed else [star_x[0], star_y[0]]
    
    if(print_log):
        if(fig is None or ax is None):
            fig, ax = plt.subplots()

        display(image_array, fig=fig, ax=ax, cmap='Greys_r')
        
        ax.scatter(star[0], star[1], color='none', edgecolor='r', s=20) # Plot trial location with an ``x``
        if(is_trimmed): # If the image is to be trimmed, plot a box to indicate search area.
            box = Rectangle((star_pos[0]-search//2, star_pos[1]-search//2), width=search, height=search, ec='g', color='none')
            ax.add_patch(box)
            ax.scatter(star_pos[0], star_pos[1], color='g', marker='x', s=20)

    return np.array(star)


def display3D(image_array, cmap=None, stretch='log', log_a = 1000, xlim = None, ylim = None, plot_view_angle=[25,90], show_colorbar=True, smooth=False, smooth_n=4, fig=None, ax=None):
    """
    **EXPERIMENTAL:** Display 2D scalar data as a 3D image.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    cmap: str, default: "Greys_r"
        A string containing the colormap title. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.

    stretch: "log" or None, default: "log"
        A string describing the type of stretching that can be applied to display image data. By default, a log stretch is applied. # TODO: Add more stretches.

    log_a: float, default: 1000
        The log index for ``stretch='log'``. # TODO: Implement log index stretching.

    xlim, ylim: float, default: None 
        Set the ``x`` or ``y`` limits of the plot. First element is the lower limit, and the second is the upper limit.

    plot_view_angle: [float, float], default: [25,90]
        Angle of view of the 3D plot.

    show_colorbar: bool, default: True
        Boolean option to show a colorbar on the 3D plot.

    smooth: bool, default: False
        Smooth every pixel by taking the sum of the pixel values of its neighbours.

    smooth_n: int, default: 2
        Number of neighbours to average over when smoothing.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    matplotlib.image.AxesImage (A plot).

    Raises
    ------
    ValueError
        If the ``stretch`` provided is not one of the allowed stretches.

    Usage
    -----
    >>> display3D(image_data, cmap="inferno", stretch='log', log_a=10, xlim=None, ylim=None, fig=None, ax=None)
    """
    allowed_stretches = ["log", "linear"]

    if(smooth):                           # If image must be smoothed:
        Ly, Lx = np.shape(image_array)    # Get image dimensions
        L = np.min([Lx,Ly])               # Find the smallest dimension

        avg = np.zeros((L-smooth_n+1, L-smooth_n+1), dtype=np.float32)

        for i in range(L-smooth_n+1):     # Loop over image and perform a
            for j in range(L-smooth_n+1): # moving average
                new_array_val = 0
                for p in range(smooth_n):
                    for q in range(smooth_n):
                        new_array_val += image_array[i+p,j+q]

                avg[i,j] = new_array_val/(smooth_n**2)

        image_array = avg                 # Reset the image array with avg

    if(stretch not in allowed_stretches):
        raise ValueError("The chosen stretch \""+ str(stretch) +"\" is not implemented. The allowed values of stretches are: "+str(allowed_stretches))

    if(fig is None or ax is None):                        # If no axis is provided,
        fig = plt.figure()                                # create an empty figure
        ax = fig.add_subplot(projection='3d')             # with a 3D subplot

    xcoords = np.arange(0,len(image_array[0]),1)          # Range of x-coordinates (in pixels)
    ycoords = np.arange(0,len(image_array[:,0]),1)        # Range of y-coordinates (in pixels) 
    
    x, y = np.meshgrid(xcoords, ycoords)                  # Create a 2D mesh-grid
    
    if(stretch=="log"):
        ax.set_zscale("log")
    elif(stretch=="linear"):
        image_array = image_array
    else:
        raise ValueError("The chosen stretch \""+ str(stretch) +"\" is not implemented. The allowed values of stretches are: "+str(allowed_stretches)+".")

    surface = ax.plot_surface(x,y, image_array,cmap=cmap) # Plot the result
    ax.view_init(plot_view_angle[0], plot_view_angle[1])  # Set the plot view angle

    ax.set_xlim(xlim)                                     # Set the ``xlim``
    ax.set_ylim(ylim)                                     # Set the ``ylim``

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_zlabel("Counts")

    if(show_colorbar):
        fig.colorbar(surface, shrink=0.75)


def sort_astrophotos(base_dir, object_prefix, symlink=True, ext="fit",  flat_prefix="flats", filter_suffixes=["L", "R", "G", "B", "Ha", "SII", "OIII"], light_suffix="", dark_suffix="D", flat_suffix="", bias_suffix="Bias", folders=["lights", "darks", "flats", "biases"], print_log=False, log_level=0):
    """
    Sort astrophotography images into appropriately named folders so that they can be used by `Siril <https://siril.org/>`__ or other software.

    This function can be used to sort images and calibration frames obtained from an automated astrophotography cameras. The files are sorted, based on their prefixes and suffixes, into appropriately named folders.

    An object is specified using its filename's prefix. The function then creates individual folders for each filter. Within each of these folders,
    separate folders for lights, darks, flats, and biases are created, within which the appropriate files are placed. Users have the option to link
    the files symbolically within these folders (strongly recommended, as it keeps the original file structure intact) or actually move them there, which would change the original file structure.

    .. important:: Windows users will have to enable **developer mode** on their machines in order to use this function, since it relies on creating symbolic links between files. Developer mode can be activated (at least on Windows 11) by going to ``Settings -> System -> For developers`` and toggling the ``Developer mode`` option.

    .. danger:: Setting ``symlink=False`` can move files around on your machine in ways that cannot easily be undone. Only use this option if you're sure you know what you're doing.

    Parameters
    ----------
    base_dir: str
        A string pointing to the path of the folder within which all the astrophotography images have been placed. This path can be either absolute or relative. Relative paths are converted to absolute paths.

    object_prefix: str
        Prefix used to filter out files for a single astronomical object. This is expected to be the first few characters of the filenames of the object's images.

    symlink: bool, default: True
        Option to symbolically link the files rather than actually moving them to the appropriate folders. This is *strongly recommended*. If ``symlink=True``, the external file structure is unchanged, only shortcuts are placed inside the newly created folders. If, on the other hand, ``symlink=False``, a new ``flats`` folder is created in the base directory, and all flats are moved into it, inside appropriately named folders per filter. The darks and biases are moved into new directories within the object's folder. These images are then symbolically linked to the appropriate folders for each filter. The lights are moved out from the external folder into the appropriate internal ``lights`` folder.

    ext: str, default: "fit"
        A string with the filename extension for the images. By default, it accepts FIT files.

    flat_prefix:, str, default: "flats"
        A string containing the prefix of the flat files. By default, these files are assumed to be named ``flats...``.

    filter_suffixes: list of str, default: ["L", "R", "G", "B", "Ha", "SII", "OIII"]
        A list of filter suffixes, assumed to be the end of the filename (before the extension).

    light_suffix: str, default: ""
        A string containing the suffix of the light files. By default, no suffix is assumed.

    dark_suffix: str, default: "D"
        A string containing the suffix for the darks. By default, "D" is used.

    flat_suffix: str, default: ""
        A string containing the suffix of the flat files. This can be used to differentiate between different flat frames taken on the same night, since this function filters out flats by looking for those filenames that follow the ``...{flat_prefix}*{flat_suffix}...`` format. By default, no suffix is assumed.

    bias_suffix: str, default: "Bias"
        A string containing the suffix of the bias files. By default, "Bias" is used.

    folders: list of str, default: ["lights", "darks", "flats", "biases"]
        A list of four strings containing the names of the different folders to be created. By default, these four folders are created for each of the filters.

    print_log: bool, default: False
        Provides the option to print a log to debug your code. In this case, it will print out which folders have been created.

    log_level: int, default: 0
        Provides the option to print out a more detailed log, indicating which files have been moved or symbolically linked. A higher level indicates a more detailed log.

    Returns
    -------
    NoneType

    Warns
    -----
    UserWarning
        If files for any of the calibration frames of a filter are not present, but the lights for that filter *are* present.

    Usage
    -----
    >>> sort_astrophotos(base_dir="./my_ast1080_folder/session5/", object_prefix="DS_M13", flats_prefix="flats", flats_suffix="FR")
    """
    def create_folder(name, print_log=False):
        ''' Helper function to create a folder using the pathlib module.'''
        Path(name).mkdir(parents=True, exist_ok=True)      # Create a folder and its parents if they don't exist
        if(print_log): print(f"Creating folder: {name}")

    def move_list(file_list, target_folder, symlink=False, print_log=False, log_level=0):
        ''' Helper function to move a list of files to a target folder, and return the new names of the moved files.'''
        new_file_list = []               # Empty list to hold the new filenames

        if(print_log): print("*"*100)    # Print some *s to make the output look a little neat

        for file in file_list:           # For every file in the filelist
            pre_move_name = file
            parentdir = ospath.dirname(pre_move_name) # Extract the parent directory
            post_move_name = pre_move_name.replace(parentdir, target_folder) # Replace the parent directory with the target name

            new_file_list.append(post_move_name) # Save the new name to the new file list

            if(symlink): # If a symbolic link is to be created
                try:
                    ossymlink(pre_move_name, post_move_name)  # Create the symbolic link
                except FileExistsError:                        # If a link already exists at the location
                    osunlink(post_move_name)                  # Unlink the old link and link the new one
                    warnings.warn(f"Symlink {post_move_name} already exists. Replacing it.", UserWarning) # Raise a warning about this.
                    ossymlink(pre_move_name, post_move_name)
                if(print_log and log_level>0): print(f"Created shortcut to {pre_move_name} from {post_move_name}.")
            else:        # If the files are not to be symlinked, then move them to the appropriate folder
                Path(pre_move_name).rename(post_move_name)
                if(print_log and log_level>0): print(f"Moved {pre_move_name} to {post_move_name}.")

        if(print_log and log_level>0): print("*"*100) # Print more stars

        return new_file_list

    # Get the absolute path of the base folder
    base_path = ospath.abspath(base_dir)

    # Load the folder names
    lights_folder, darks_folder, flats_folder, biases_folder = folders

    # Load the dark file list
    dark_files = get_files(f"{base_path}/{object_prefix}*{dark_suffix}.{ext}")

    # Load the bias file list
    bias_files = get_files(f"{base_path}/{object_prefix}*{bias_suffix}.{ext}")

    # Load the lights file list (one list per filter)
    lights_files = [get_files(f"{base_path}/{object_prefix}*{light_suffix}*{filter}.{ext}")             for filter in filter_suffixes]

    # Load the flats file list  (one list per filter)
    flats_files  = [get_files(f"{base_path}/{flat_prefix}*{flat_suffix}*{filter}.{ext}") for filter in filter_suffixes]
    # Load the flats calibration files (just in case it's needed later)
    flats_cal_files  = [get_files(f"{base_path}/{flat_prefix}*{flat_suffix}*{filter}.{ext}") for filter in [dark_suffix, bias_suffix]]

    if(symlink is False): # If symbolic links are not to be created, the folder structure is different:
        # - A new folder is created in the parent directory within which all the flats reside, in subfolders for each filter.
        # - The darks and the biases for each object are moved and placed within the object's folder.
        # - Eventually, symbolic links are created between these files and the files per filter created further down.

        # For simplicity, a list of calibration folders and files is created
        cal_folders = [f"{object_prefix}/{darks_folder}", f"{object_prefix}/{biases_folder}"] + ([f"{flats_folder}/{filter}" for filter in filter_suffixes ]) + ([f"{flats_folder}/{filter}" for filter in [dark_suffix, bias_suffix] ])
        cal_files   = [dark_files, bias_files] + ([flats for flats in flats_files]) + ([flats for flats in flats_cal_files])

        for c in range(len(cal_folders)):                         # For each calibration frametype,
            base_folder = f"{base_path}/{cal_folders[c]}"         # define a new appropriately named folder
            if(len(cal_files[c])>0):                              # if there are any files of that frame,
                create_folder(base_folder, print_log=print_log)   # and move the files there, returning the new names
                cal_files[c] = move_list(cal_files[c], target_folder=base_folder, symlink=symlink, print_log=print_log, log_level=log_level)
            else:
                cal_files[c] = get_files(f"{base_folder}/*.{ext}")

        dark_files, bias_files = cal_files[0], cal_files[1]       # Replace the old calibration frame list
        flats_files = cal_files[2:]                               # with the new (replaced) calibration frame list

    for f in range(len(filter_suffixes)):                         # Loop over every filter
        filter = filter_suffixes[f]

        # For simplicity again, we define lists containing all the source files and target folders for each filter
        source_files    = [lights_files[f], dark_files, flats_files[f], bias_files]
        target_folders  = [lights_folder, darks_folder,flats_folder,  biases_folder]

        # We define a boolean variable to decide whether files should be linked symbolically or not.
        # By default, lights are linked symbolically. If ``symlink=False`` then lights are moved.
        # Note that within each filter's folder the darks, biases, and flats are all symlinked always.
        symlink_files = [symlink, True, True, True]

        for i in range(len(target_folders)):  # Create the lights, darks, flats, and biases folders
            if(len(source_files[0])>0):       # (Only do this if there are actually any lights to place in the folders)
                if(len(source_files[i])==0):
                    warnings.warn(f"No {target_folders[i]} for filter \"{filter}\". Are you sure you've chosen your names correctly?", UserWarning)
                target = f"{base_path}/{object_prefix}/{filter}/{target_folders[i]}"
                create_folder(target, print_log=print_log)
                move_list(file_list=source_files[i], target_folder=target, symlink=symlink_files[i], print_log=print_log, log_level=log_level)
