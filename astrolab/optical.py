#!/usr/bin/env python

"""
This package contains a list of functions that can be used to analyse images. This includes images taken for astrophotography, grating spectroscopy, and stellar photometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from astropy.io import fits
from astropy.visualization import simple_norm

import warnings
from PIL import Image

from matplotlib.patches import Rectangle


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
        A boolean variable to decide whether you want to print a log of what you've done. In this case, whether you want display the image you have loaded.

    cmap: str, default: "Greys_r"
        A string containing the colormap title. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.

    stretch: "log" or None, default: "log"
        A string describing the type of stretching that can be applied to display image data. By default, a log stretch is applied. # TODO: Add more stretches

    log_a: float, default: 1000
        The log index for ``stretch='log'``. # TODO: Implement log index stretching

    Returns
    -------
    data: array_like
        A 2D array containing the image data.

    Raises
    ------
    ValueError
        If the filetype is not FIT, FITS, JPG, JPEG, or PNG
    
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


def display(image_array, cmap='Greys_r', stretch='log', log_a = 1000, norm_array=None, xlim = None, ylim = None, fig=None, ax=None):
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

    xlim, ylim: float, default: None 
        Set the ``x`` or ``y`` limits of the plot. First element is the lower limit, and the second is the upper limit.

    norm_array: array_like, default: None
        A 2D array used to decide the normalisation of the ``simple_norm`` (from ``astropy.visualization``) used to visualise the plot. By default, the ``image_array`` is used for this norm.

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
        
    img = ax.imshow(image_array, origin='lower', cmap = cmap, norm=simple_norm(norm_array, stretch=stretch, log_a = log_a))        # Display the image

    ax.set_xlim(xlim)                                     # Set the ``xlim`` of the plot
    ax.set_ylim(ylim)                                     # Set the ``ylim`` of the plot

    return img


def display3D(image_array, cmap=None, stretch='log', log_a = 1000, xlim = None, ylim = None, plot_view_angle=[25,90], fig=None, ax=None):
    """
    EXPERIMENTAL: Display 2D scalar data as a 3D image.

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
    >>> plot3D(image_data, cmap="inferno", stretch='log', log_a=10, xlim=None, ylim=None, fig=None, ax=None)
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
    
    WARNING: Shifting an image may cause parts of the image array to be cut off.

    Parameters
    ----------
    array_of_images: array_like 
        A 3-dimensional ``(N, Ly, Lx)`` array containing ``N`` image arrays of size ``(Ly, Lx)`` to stack. 

    shifts: array_like or None, default: None
        An (N, 2) array containing N 2D vectors representing the shifts by which each image should be displaced before summing. If ``None`` is provided, no shifting is done.

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


def find_angle(image_array, threshold=0.1, star=[None, None], search=500, print_log=False, fig=None, ax=None):
    """
    Find angle by which to rotate an image with a spectrum so that the spectrum is horizontal.

    The function works by fitting a straight line to the brightest points in the pixel, and finding its slope. The arctangent of this slope is used to find the angle by which the image must be rotated.

    **NOTE:** The algorithm assumes the spectrum faces rightward. If this is not the case, first flip the image using the ``flip`` function.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    threshold: float < 1, default: 0.1
        Threshold value (as a fraction of the maximum value in the image) below which all data-points are ignored.

    star: [int, int] or [None, None], default: [None, None]
        `x` and `y` pixel coordinates of the star's rough location to refine search for the brightest pixel. # TODO: Incorporate this. If None is provided, the brightest pixel is used.

    search: float, default: 500
        Search "radius" in pixels. The brightest pixel will be found in the range (star_pos[0] +/- search, star_pos[1]+/- search).

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it will plot the data-points above the threshold, and the trendline that is fit through it.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    angle: float 
        Value of angle by which the image is to be rotated.

    Usage
    -----
    >>> this_angle = find_angle(this_data, threshold=0.22)
    """
    data_y, data_x = np.where(image_array > np.max(image_array)*threshold) # Get data above the threshold
    
    slope, intercept = np.polyfit(data_x, data_y, deg=1) # Fit a straight line to this data 
    angle = np.arctan(slope)*180/np.pi                   # Find angle using the slope of the trendline

    if(print_log):
        if(fig is None or ax is None):
            fig, ax = plt.subplots()
        ax.scatter(data_x, data_y, edgecolor='k', color='none')  # Plot thresholded data
        ax.plot(data_x, slope*data_x + intercept, color='firebrick', ls='--') # Plot trendline

        xlims = ax.get_xlim(); ylims = ax.get_ylim() # Get the x and y limits to print angle
        x0 = xlims[0] + (xlims[1]-xlims[0])/2
        ax.text(1.05*x0, slope*x0 + intercept, "Angle = "+str(np.round(angle,2))+r"$^\circ$", color='firebrick') # Put some text

    return angle


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
    
        
def shift(image_array, displacement, print_log=False):
    """
    Translate an image by a ``displacement`` vector.
    
    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    displacement: [float, float] 
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
    

def rotate(image_array, angle=None, star=[None,None], threshold=0.1, expand=False, fill=False, print_log=False):
    """
    Rotate an image around a star. If no ``angle`` is provided, the angle is computed using ``find_angle``. The star location can be found as the output of the ``find_star`` function. If no star location is provided, it simply finds the brightest pixel.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    angle: float or None, default: None
        Angle by which to rotate the image.

    star: [int, int] or [None, None], default: [None, None]
        ``x`` and ``y`` pixel coordinates of the star's exact location.
    
    threshold: float < 1, default: 0.1
        Threshold value (as a fraction of the maximum value in the image) below which all data-points are ignored.

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
    >>> rotated_array = rotate(this_data, angle=23.0, star=[1052,1002])
    >>> rotated_array = rotate(this_data, star=[1052,1002], threshold=0.22)
    
    """

    if(star[0] is None or star[1] is None):
        star = find_star(image_array, print_log=print_log)

    if(angle is None):
        angle = find_angle(image_array, threshold=threshold, print_log=print_log)

    peak = np.max(image_array)
    image   = Image.fromarray(image_array/peak) # TODO: This is done to avoid type-casting issues. Needs to be fixed.
    fillcolor= np.median(image_array/peak) if fill else None # Fill expanded region with the median value
    r_image = image.rotate(angle, resample=Image.BICUBIC, center=star, expand=expand, fillcolor=fillcolor) # Rotate image about star position
    rotated_array = np.asarray(peak*r_image)

    if(print_log):
        display(rotated_array, norm_array=image_array)
    
    return rotated_array


def crop(image_array, star = [None, None], offset = 100, width = 1500, height = 200, print_log=False, cmap='Greys_r'):
    """
    Crop an image around a star, with ``offset`` space to the left, a total width of ``width`` and height ``height``. The star location can be found using the ``find_star`` function. If no star location is provided, it simply finds the brightest pixel.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.
        
    star: [int, int] or [None, None], default: [None, None] 
        ``x`` and ``y`` pixel coordinates of the star's exact location.
    
    offset: int, default: 100 
        Space to the left of star in cropped image.

    width: int, default: 1500 
        Width of the cropped image.

    height: int, default: 200. 
        Height of the cropped image.

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it displays the cropped array.

    cmap: str, default: "Greys_r" 
        A string containing the colormap title if ``print_log=True`` and a plot is produced. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.
    
    Returns
    -------
    cropped_array: array_like
        A 2D array. The flipped array.

    Raises
    ------
    ValueError
        If the offset is larger than half the image dimension in either direction.
        
    Warns
    -----
    UserWarning
        If ``width`` is larger than the image's width.

    UserWarning
        If ``height`` is larger than the image height.

    UserWarning
        If there isn't enough space between the star and the left/bottom of the image.

    Usage
    -----
    >>> cropped_array = crop(this_data, star=[1052,1002], offset=100, width=2500, height=400)
    """
    if( offset > len(image_array)/2 or offset > len(image_array[0])/2 ):
        raise ValueError("Offset value " + str(offset) + " cannot be larger than half either image dimension " + str(np.shape(image_array)) + ".")
    
    if(width > len(image_array[0])):
        warnings.warn("`width` cannot be greater than image width, resizing width to image width.", UserWarning)
        width = len(image_array[0])
        
    if(height > len(image_array)):
        warnings.warn("`height` cannot be greater than image height, resizing height to image height.", UserWarning)
        width = len(image_array)
        
    if(star[0]==None or star[1]==None): # If star is not provided, find the brightest pixel
        star = find_star(image_array)
    
    offsets = np.array([offset, height//2])
    x,y = 0, 1

    for d in x,y: # Check to see if there is enough space between the star and the left/bottom of the image
        dir = "left" if d==0 else "bottom"
        if(star[d] - offsets[d] < 0):
            offsets[d] = star[d]
            warnings.warn("Not enough space to the "+dir+"; offset changed to "+ str(offsets[d])+".", UserWarning)

        # TODO: What if the star is too close the right/top of the image?
    
    crop_x_range = [star[x] - offsets[x], star[x] - offsets[x] + width ] # x range (in pixels) to crop
    crop_y_range = [star[y] - offsets[y], star[y] - offsets[y] + height] # y range (in pixels) to crop
    
    cropped_array = image_array[crop_y_range[0] : crop_y_range[1], crop_x_range[0] : crop_x_range[1]]

    if(print_log):
        fig, ax = plt.subplots()
        display(cropped_array, fig=fig, ax=ax, cmap=cmap)

    return cropped_array
    

def get_spectrum(image_array, sub_bkg=False, lower_lim=None, upper_lim=None, n_sigma = 3, print_log=False, fig=None, ax=None):
    """
    Produce a spectrum from a cropped image, with the option to compute and subtract the background.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    sub_bkg: bool, default: False
        Compute and subtract background from spectrum.

    lower_lim: int or None, default: None
        Lower row limit of spectrum. All pixel rows below this are taken to be background.

    upper_lim: int or None, default: None
        Upper row limit of spectrum. All pixel rows above this are taken to be background.

    n_sigma: float, default: 3.0
        Width of the spectrum beyond which background can be taken.

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it plots the point-spread function of the star if ``n_sigma`` is not ``None``, or the lower and upper limits if those are provided. It also plots the spectrum before and after background subtraction.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    spectrum: array_like
        A 1D array with information of the intensity as a function of pixel.

    Usage
    -----
    >>> this_spectrum = get_spectrum(cropped_array, n_sigma=4)
    """
    bkg = np.zeros_like(image_array[0])      # Initial row of zeros to store background
    spectrum = np.mean(image_array, axis=0)  # Vertical average of image to produce spectrum
    
    if(sub_bkg):                             # If background is to be subtracted
        if(lower_lim is None or upper_lim is None):
            # TODO: Implement this better.
            vertical_profile = np.mean(image_array, axis=1) # first plot vertical profile of data,
            maxval = np.where(vertical_profile == np.max(vertical_profile)) # find value at which profile is maximum,
            std = np.std(vertical_profile) # find sigma of this data 
        
            lower_lim = int(maxval - n_sigma*std) # Find number of lines below (integer)
            upper_lim = int(maxval + n_sigma*std) # Find number of lines above (integer)
            
            if(print_log):
                if(fig is None or ax is None):
                    fig, ax = plt.subplots()
        
                ax.plot(vertical_profile, color='k', label="Vertical profile")
                ax.axvline(lower_lim, color='firebrick', label="Cutoffs")
                ax.axvline(upper_lim, color='firebrick')
                ax.legend()
                
        elif(print_log):
            if(fig is None or ax is None):
                fig, ax = plt.subplots()
    
            display(image_array, fig=fig, ax=ax)
            ax.axhline(lower_lim, color='firebrick', label="Cutoffs")
            ax.axhline(upper_lim, color='firebrick')
            ax.legend()
        
        lower_bkg = image_array[:lower_lim]  # Selection of lines for lower background
        upper_bkg = image_array[upper_lim:]  # Selection of lines for upper background

        bkg  = (np.mean(lower_bkg, axis=0) + np.mean(upper_bkg, axis=0))/2 # Average background

        spectrum = np.mean( image_array[lower_lim:upper_lim], axis=0 ) # Recompute spectrum ignoring background
        
    if(print_log):
        fig, ax = plt.subplots()
        ax.plot(spectrum, lw=1, color='firebrick', label='No background subtraction')
        if(sub_bkg):
            ax.plot(spectrum-bkg, lw=1, color='steelblue', label='Background subtracted')
        ax.legend()

    return spectrum-bkg


def plot_ref(wvs = [6562.79, 4861.35, 4340.472, 4101.734], wvnames=[r"$\alpha$",r"$\beta$",r"$\gamma$", r"$\delta$"], color='darkgoldenrod', lw=0.5, text_rotation=90, fig=None, ax=None):
    """
    Plot vertical lines of a reference spectrum. By default, the Balmer lines are plotted in Angstroms.

    Parameters
    ----------
    wvs: array_like, default: [6562.79, 4861.35, 4340.472, 4101.734] (Balmer series)
        A list of wavelengths.

    wvnames: array_like, default: [r"$\alpha$",r"$\beta$",r"$\gamma$", r"$\delta$"] (Balmer series)
        A list of wavelength names.

    color: str, default: 'darkgoldenrod'
        Colour of this plot. Must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    lw: float, default: 0.5
        Linewidth for the vertical reference lines.
        
    text_rotation: float, default: 90
        Angle in degrees by which to rotate text labels.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.
    
    Returns
    -------
    matplotlib.image.AxesImage (A plot).

    Usage
    -----
    >>> plot_ref(this_spectrum, fig=fig, ax=ax)
    """
    if(fig==None or ax==None):
        fig, ax = plt.subplots()

    ylims = ax.get_ylim()      # Get the y-lim, to position the text. TODO: Implement this better.
    
    for w in range(len(wvs)):  # For each wavelength in the list, plot a vertical line
        ax.axvline(wvs[w], color=color, lw=lw)
        ax.text(wvs[w]*1.005, ylims[1] - 0.1*(ylims[1]-ylims[0]), wvnames[w], color=color, rotation=text_rotation)

def plot_fraunhofer(wvs = [7589.0, 6865.0, 6555.0, 5895.0, 5273.0, 5185.0, 4876.0, 4321.0], wvnames=["A", "B", "C", r"D$_1$", r"E$_1$", "b$_1$", "F", "G"], color='firebrick',lw=0.5, text_rotation=90, fig=None, ax=None):
    """
    Plot vertical lines of a reference spectrum. By default, the Balmer lines are plotted in Angstroms.

    NOTE: This is just a wrapper function around the ``plot_ref`` function.

    Parameters
    ----------
    wvs: array_like, default: [7589.0, 6865.0, 6555.0, 5895.0, 5273.0, 5185.0, 4876.0, 4321.0] (Fraunhofer lines).
        A list of wavelengths.

    wvnames: array_like, default: ["A", "B", "C", r"D$_1$", r"E$_1$", "b$_1$", "F", "G"] (Fraunhofer lines)
        A list of wavelength names.

    color: str, default: 'darkgoldenrod'
        Colour of this plot. Must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    lw: float, default: 0.5
        Linewidth for the vertical reference lines.
        
    text_rotation: float, default: 90
        Angle in degrees by which to rotate text labels.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.
    
    Returns
    -------
    matplotlib.image.AxesImage (A plot).

    Usage
    -----
    >>> plot_fraunhofer(this_spectrum, fig=fig, ax=ax)
    """

    plot_ref(wvs=wvs, wvnames=wvnames, color=color, fig=fig, ax=ax, lw=lw, text_rotation=text_rotation)
    

def calibrate(spectrum, lineA, lineB=None, wvPerPix=None, print_log=False, lw=0.5, xlim=[None,None], ylim=[None,None], fig=None, ax=None):
    """
    Calibrate a spectrum using either one-point or two-point calibration, assuming the relation between wavelength and pixel is linear. 
    
    If only the spectrum and details of one point are given, one-point calibration is done using the value of ``wvPerPix`` provided. 
    
    If the data for two points are given, ``wvPerPix`` is calculated using both points.

    Parameters
    ----------
    spectrum: array_list
        An array of spectrum values.

    lineA: list [int, float] 
        First element (integer) is the pixel number, second element (float) is the wavelength.

    lineB: list [int, float] or None, default: None 
        Required for two-point calibration. First element is the pixel number, second element is the wavelength.

    wvPerPix: float or None, default: None 
        Required for one-point calibration. Number of wavelength units (angstroms or nanometres) per pixel. If two-point calibration is done, then this quantity is computed and returned. If both ``lineA`` and ``lineB`` is provided in addition to ``wvPerPix``, ``wvPerPix`` is ignored and two-point calibration is done.

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it plots the calibrated spectrum along with vertical lines to mark ``lineA`` and (if provided) ``lineB``.

    lw: float, default 0.5
        Linewidth for the vertical calibration lines.

    xlim, ylim: [float, float] or [None, None], default: [None, None]
        Set the ``x`` or ``y`` limits of the plot. First element is the lower limit, and the second is the upper limit.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    
    Returns
    -------
    wavelengths: array_like
        An array of wavelengths for every pixel value.
    wvPerPix: float
        Value of number of wavelength units per pixel. IMPORTANT: Only returned if 2-point calibration is being done.

    Raises
    ------
    ValueError
        If ``lineA`` is not provided.

    ValueError
        If ``lineB`` is not provided **and** ``wvPerPix`` is not provided.

    Warns
    -----
    UserWarning
        If both ``lineA`` and ``lineB`` are provided **and** ``wvPerPix`` is provided. In this case, ``wvPerPix`` is ignored.

    Usage
    -----
    >>> this_wvs, this_wvPerPix = calibrate(this_spectrum, lineA=[100,0], lineB=[767,486.135]) # For two-point calibration
    
    >>> this_wvs = calibrate(this_spectrum, lineA=[100,0], wvPerPix=0.48) # For 1-point calibration
    """
    if(lineA is None):
        raise ValueError("One calibration point always needed, please provide `lineA`.")
     
    if(lineB is None and wvPerPix is None):
        raise ValueError("You must either provide `wvPerPix` (for one-point calibration), or `lineB` (for two-point calibration).")

    if(lineB!=None and wvPerPix!=None):
        warnings.warn("Two-point information provided, ignoring `wvPerPix` for calibration.", UserWarning)

    
    pixels = np.arange(0, len(spectrum))   # List of pixels
    
    if(lineB==None):                       # If no second point is given, one-point calibration is done.
        pA = lineA[0]; lA = lineA[1];
        
        offsetPix = lA-wvPerPix*pA         # Find intercept from `wvPerPix`.
        wavelengths = wvPerPix * pixels + offsetPix  # Get array of wavelengths

    else:                                  # Do two-point calibration
        pA = int(lineA[0]); lA = lineA[1]; 
        pB = int(lineB[0]); lB = lineB[1];
    
        wvPerPix = (lB-lA)/(pB-pA)         # Find slope
        offsetPix = ((lB + lA) - wvPerPix * (pB + pA))/2 # Find intercept
    
        wavelengths = wvPerPix * pixels + offsetPix # Get array of wavelengths

    if(print_log and (fig==None or ax==None)):
        fig, ax = plt.subplots()

        ax.plot(pixels, spectrum, color='k')
        ax.axvline(pA, color='firebrick', lw=lw)
        if(lineB!=None):
            ax.axvline(pB, color='firebrick', lw=lw)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    if(lineB!=None):               # If two-point calibration is done, return the `wvPerPix` as well
        return wavelengths, wvPerPix
    
    return wavelengths             # For one-point calibration just return the wavelengths