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

from matplotlib.patches import Rectangle
import glob as glob


def load_image(filename, gray_conv = [0.2126, 0.7152, 0.0722], print_log=False, cmap='Greys_r', stretch='log', log_a=1000):
    """
    Load an image by filename and produce a 2D array. 
    
    Input images can either be fits or jpg images. If jpg images are used, they are converted to grayscale first through a grayscale conversion vector that can be modified.

    Parameters
    ----------
    filename: str
        The image file to read: a filename pointing to the location of the file.

    gray_conv: [float, float, float], default: [0.2126, 0.7152, 0.0722] 
        R,G, and B weights to convert from RGB image to grayscale image. 

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, whether you want display the image you have loaded.

    cmap: str, default: "Greys_r"
        A string containing the colormap title. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.

    stretch: "log" or None, default: "log"
        A string describing the type of stretching that can be applied to display image data. By default, a log stretch is applied. # TODO: Add more stretches

    log_a: float, default: 1000
        The log index for ``stretch='log'``.

    Returns
    -------
    data: array_like
        A 2D array containing the image data.

    Raises
    ------
    ValueError
        If the filetype is not FIT, FITS, JPG, JPEG, or PNG.
    
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
    else:
        raise ValueError("Filetype not supported: "+extension )

    if(print_log):                                          # Display image, if a log is requested
        display(data, cmap=cmap, stretch=stretch, log_a=log_a)

    return data.astype(np.float32)


def display(image_array, cmap='Greys_r', stretch='log', log_a = 1000, norm_array=None, min_percent=0.0, max_percent=100.0, title=None, xlim = None, ylim = None, fig=None, ax=None):
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
        Provides option to print a log to debug your code. In this case, it will print out the list of files loaded.

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


def stack_files(filelist, stack_type='mean', gray_conv = [0.2126, 0.7152, 0.0722], print_log=False):
    """
    Stack multiple image files together to produce a single output file. Input files can either
    be a list of fits or jpg images. If jpg images are used, they are converted to grayscale first using the ``gray_conv`` parameter.

    Parameters
    ----------
    filelist: array_like
        List of strings containing the filenames of files that need to be added.

    stack_type: {"mean", "average", "sum"}, default: "mean"
        A string describing the type of stacking. "average" and "mean" both perform the same action.

    print_log: bool, default: False
        A boolean variable to decide whether you want to print a log of what you've done. In this case, a list of the files stacked.

    Returns
    -------
    stacked: array_like
        A 2D array containing the sum (or average) of every item in the file-list.

    Raises
    ------
    ValueError
        If ``stack_type`` is not one of the available types.

    Usage
    -----
    >>> stacked_image = stack_files(filelist, stack_type='mean')
    """
    stacked = np.zeros_like(load_image(filelist[0]), dtype=np.float32) # Create an empty array of zeros
    
    for i in range(len(filelist)):                              # Load each file,
        stacked += load_image(filelist[i], gray_conv=gray_conv, print_log=print_log) # and add it to ``stacked``
        if(print_log):
            print("File completed:", filelist[i])

    if(stack_type=='mean' or stack_type=='average'):
        stacked = stacked/len(filelist) # Divide by number of files to obtain the mean
    elif(stack_type != 'sum'):
        raise ValueError("`stack_type` should be either \"sum\" or \"mean\" (or \"average\").")

    return stacked


def stack(array_of_images, shifts = None, stack_type='mean', print_log=False):
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
    
        
def crop(image_array, left=None, right=None, top=None, bottom=None, origin=None, print_log=False):
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
        Provides option to print a log to debug your code. In this case, it will display the cropped array.

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
        display(cropped_array)

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
        Provides option to print a log to debug your code. In this case, it will display the shifted array.
    
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
        Provides option to print a log to debug your code. In this case, it will display the rotated array.
    
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


def find_star(image_array, star_pos=[None, None], search=500, print_log=False):
    """
    Find the pixel location of star in a 2D image array. A "star" is defined simply as the "brightest" pixel in the array. 
    
    If multiple bright pixels can be found, a rough location and search area can be defined to refine the search.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    star_pos: [int, int] or [None, None], default: [None, None]
        ``x`` and ``y`` pixel coordinates of the star's rough location, to refine search for the brightest pixel. If None is provided, the brightest pixel is used.

    search: float, default: 500
        Search "radius" in pixels. The brightest pixel will be found in the range (star_pos[0] +/- search, star_pos[1]+/- search).

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it will show the location of the detected star, and the provided search-box.

    Returns
    -------
    star: array_like
        An array of 2 elements containing ``x`` and ``y`` pixel location of the "star".

    Usage
    -----
    >>> this_star = find_star(this_data, star_pos=[1000,1500], search=500)
    """
    if(star_pos[0] is not None and star_pos[1] is not None):  # If both ``x`` and ``y`` coordinates are provided
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
        fig, ax = plt.subplots()
        display(image_array, fig=fig, ax=ax, cmap='Greys_r')
        
        ax.scatter(star[0], star[1], color='none', edgecolor='r', s=20) # Plot trial location with an ``x``
        if(is_trimmed): # If the image is to be trimmed, plot a box to indicate search area.
            box = Rectangle((star_pos[0]-search//2, star_pos[1]-search//2), width=search, height=search, ec='g', color='none')
            ax.add_patch(box)
            ax.scatter(star_pos[0], star_pos[1], color='g', marker='x', s=20)

    return np.array(star)


def display3D(image_array, cmap=None, stretch='log', log_a = 1000, xlim = None, ylim = None, plot_view_angle=[25,90], fig=None, ax=None):
    """
    **EXPERIMENTAL:** Display 2D scalar data as a 3D image.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    cmap: str, default: "Greys_r"
        A string containing the colormap title. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.

    stretch: "log" or None, default: "log"
        A string describing the type of stretching that can be applied to display image data. By default, a log stretch is applied. # TODO: Add more stretches

    log_a: float, default: 1000
        The log index for ``stretch='log'``. # TODO: Implement log index stretching

    xlim, ylim: float, default: None 
        Set the ``x`` or ``y`` limits of the plot. First element is the lower limit, and the second is the upper limit.

    plot_view_angle: [float, float], default: [25,90]
        Angle of view of the 3D plot.

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
    allowed_stretches = ["log"]

    if(stretch not in allowed_stretches):
        raise ValueError("The chosen stretch \""+ str(stretch) +"\" is not implemented. The allowed values of stretches are: "+str(allowed_stretches))

    if(fig is None or ax is None):                        # If no axis is provided,
        fig = plt.figure()                                # create an empty figure
        ax = fig.add_subplot(projection='3d')             # with a 3D subplot

    xcoords = np.arange(0,len(image_array[0]),1)          # Range of x-coordinates (in pixels)
    ycoords = np.arange(0,len(image_array[:,0]),1)        # Range of y-coordinates (in pixels) 
    
    x, y = np.meshgrid(xcoords, ycoords)                  # Create a 2D mesh-grid
    
    if(stretch is not None):
        image_array = np.log(image_array)

    ax.plot_surface(x,y, image_array, cmap=cmap)          # Plot the result
    ax.view_init(plot_view_angle[0], plot_view_angle[1])  # Set the plot view angle

    ax.set_xlim(xlim)                                     # Set the ``xlim`` of the plot
    ax.set_ylim(ylim)                                     # Set the ``ylim`` of the plot

