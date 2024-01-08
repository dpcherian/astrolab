#!/usr/bin/env python

"""
This package contains a list of functions specifically designed to analyse images with spectra in them. These functions are intended to be used in the Fraunhofer Lines and Grating Spectroscopy experiments.
"""

import numpy as np
import matplotlib.pyplot as plt

import warnings

import astrolab.imaging


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


def rotate_spectrum(image_array, angle=None, origin=None, threshold=0.1, expand=False, fill=False, print_log=False):
    """
    Rotate an image by an angle around an origin. If no origin is provided, it rotates about the centre of the image. If no angle is provided, the angle is computed using ``imaging.find_angle``. The origin could be the location of a star, found as the output of the ``imaging.find_star`` function.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    angle: float or None, default: None
        Angle by which to rotate the image.

    origin: [int, int] or None, default: None
        ``x`` and ``y`` pixel coordinates of the star's exact location.
    
    threshold: float < 1, default: 0.1
        Threshold value (as a fraction of the maximum value in the image) below which all data-points are ignored.

    expand: bool, default: False
        Expands the output image to make it large enough to hold the entire rotated image. Note that this flag assumes rotation about the centre, and no translation.

    fill: bool, default: False
        Fill area outside the rotated image. If ``True``, this area is filled with the median value of the image.

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it will display the rotated array.
    
    Returns
    -------
    rotated_array: array_like
        A 2D array. The rotated array.

    Warns
    -----
    UserWarning
        If ``expand=True`` and ``origin`` is provided.

    Usage
    -----
    >>> rotated_array = rotate_spectrum(this_data, angle=23.0, star=[1052,1002])
    >>> rotated_array = rotate_spectrum(this_data, star=[1052,1002], threshold=0.22)
    """
    if(origin is None):    # If no origin is provided,
        origin = [len(image_array[0])//2, len(image_array)//2] # use the closest point to the centre of the image
    elif(expand == True):  # If the ``expand`` flag is set, and the origin isn't the centre of the image, the behaviour might not be as expected
        warnings.warn("``expand=True`` assumes rotation about the centre of the image, and no translation. If you have provided any other point, the image might get truncated.", UserWarning)

    if(angle is None): # If no angle is provided,
        angle = find_angle(image_array, threshold=threshold, print_log=print_log) # find one

    rotated_array = astrolab.imaging.rotate(image_array=image_array, angle=angle, origin=origin, expand=expand, fill=fill, print_log=print_log) # Rotate using the ``imaging.rotate`` function, using ``origin`` as the origin.
    
    return rotated_array


def crop_spectrum(image_array, origin=None, offset=100, width=1500, height=200, print_log=False, cmap='Greys_r'):
    """
    Crop an image around a star, with ``offset`` pixels to the left, a total width of ``width`` and height ``height``. The star location can be found using the ``imaging.find_star`` function. If no star location is provided, it simply finds the brightest pixel.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.
        
    origin: [int, int] or None, default: None
        ``x`` and ``y`` pixel coordinates of the origin. The cropped image starts from ``offset`` pixels to the left of this point, with a total width of ``width``, and a total height of ``height`` about this point. By default, the centre of the image is chosen.
    
    offset: int, default: 100
        Space to the left of star in cropped image.

    width: int, default: 1500
        Width of the cropped image.

    height: int, default: 200
        Height of the cropped image.

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it displays the cropped array.

    cmap: str, default: "Greys_r"
        A string containing the colormap title if ``print_log=True`` and a plot is produced. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.
    
    Returns
    -------
    cropped_array: array_like
        A 2D array. The cropped array.

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
    >>> cropped_array = crop_spectrum(this_data, origin=[1052,1002], offset=100, width=2500, height=400)
    """
    if( offset > len(image_array)/2 or offset > len(image_array[0])/2 ):
        raise ValueError("Offset value " + str(offset) + " cannot be larger than half either image dimension " + str(np.shape(image_array)) + ".")
    
    if(width > len(image_array[0])):
        warnings.warn("`width` cannot be greater than image width, resizing width to image width.", UserWarning)
        width = len(image_array[0])
        
    if(height > len(image_array)):
        warnings.warn("`height` cannot be greater than image height, resizing height to image height.", UserWarning)
        height = len(image_array)
        
    if(origin is None): # If no origin is provided,
        origin = [len(image_array[0])//2, len(image_array)//2] # use the closest point to the centre of the image
    
    offsets = np.array([offset, height//2])
    x,y = 0, 1

    for d in x,y: # Check to see if there is enough space between the star and the left/bottom of the image
        dir = "left" if d==0 else "bottom"
        if(origin[d] - offsets[d] < 0):
            offsets[d] = origin[d]
            warnings.warn("Not enough space to the "+dir+"; offset changed to "+ str(offsets[d])+".", UserWarning)

        # TODO: What if the star is too close the right/top of the image?
    
    crop_x_range = [origin[x] - offsets[x], origin[x] - offsets[x] + width ] # x range (in pixels) to crop
    crop_y_range = [origin[y] - offsets[y], origin[y] - offsets[y] + height] # y range (in pixels) to crop
    
    cropped_array = image_array[crop_y_range[0] : crop_y_range[1], crop_x_range[0] : crop_x_range[1]]

    if(print_log):
        fig, ax = plt.subplots()
        astrolab.imaging.display(cropped_array, fig=fig, ax=ax, cmap=cmap)

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
    
            astrolab.imaging.display(image_array, fig=fig, ax=ax)
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
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Intensity")
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

    wvnames: array_like, default: [r"$\\alpha$",r"$\\beta$",r"$\\gamma$", r"$\\delta$"] (Balmer series)
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

    **NOTE:** This is just a wrapper function around the ``plot_ref`` function.

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

        ax.set_xlabel("Pixel")
        ax.set_ylabel("Intensity")

    if(lineB!=None):               # If two-point calibration is done, return the `wvPerPix` as well
        return wavelengths, wvPerPix
    
    return wavelengths             # For one-point calibration just return the wavelengths