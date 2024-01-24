#!/usr/bin/env python

"""
This package contains a list of functions that can be used to analyse images of stars to perform stellar photometry on them. These functions are intended to be used in the stellar photometry experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings

from pkgutil import get_data
from io import StringIO

import astrolab.imaging


def get_star_counts(image_array, star=None, radii=np.arange(20), bkg_counts_pp=0, points_to_avg=1, print_log=False, cmap='Greys_r', plot_apertures=5, aperture_color="firebrick", fig=None, axes=None):
    """
    Perform aperture photometry on a star.

    The number of counts (pixel values) are measured within a range of apertures whose radii are provided. A constant number of background counts per pixel is removed from each aperture. This number should plateau at the number of counts due solely to the star. The last few points are averaged to return this number.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    star: [int, int] or None, default: None
        `x` and `y` pixel coordinates of the star's rough location to refine search for the brightest pixel. If ``None`` is provided, the star's location is determined using the brightest pixel.

    radii: array_like, default: numpy.arange(20)
        A list of radii for the different apertures at which the star's counts are calculated. By default, a range of radii up to 20 pixels is used.

    bkg_counts_pp: float, default: 0.0
        A constant number of background counts per pixel. This number is used to compute the total counts due to the background in the aperture. If the aperture is of area `A` pixels, then the total counts due to the background for that aperture is ``bkg_counts_pp * A``. By default, this value is 0.0.

    points_to_avg: int, default: 1
        The last `n` points to average to give the star's final counts after background subtraction. By default, only the last point's value is used.

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it produces a figure with two plots: one which visually shows the star with some of the apertures overlaid, and the other which shows the number of counts as a function of aperture radius.

    cmap: str, default: "Greys_r"
        A string containing the colormap title. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.

    plot_apertures: int, default: 5
        The number of apertures to overlay over the image of the star in the log.

    aperture_color: str, default: 'firebrick'
        Colour of the apertures overlaid on the image of the star. Must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    axes: ndarray of two elements which are matplotlib axes objects, default: None
        Axes on which to plot the result. By default, a new axis of two columns is created.

    Returns
    -------
    star_counts: float
        The final averaged number of star counts.

    Usage
    -----
    >>> mizar_counts = get_distances(mizar_image, star=[1024,512])
    """

    if(star is None): # If no star is given, use the brightest pixel
        star = astrolab.imaging.find_star(image_array=image_array)

    if(print_log):
        if(fig is None or axes is None):
            fig, axes = plt.subplots(nrows=1, ncols=2, width_ratios=[2,3], figsize=(11, 3))

        astrolab.imaging.display(image_array=image_array, fig=fig, ax=axes[0], cmap=cmap)

        skip = len(radii)//plot_apertures + 1 # Number of apertures to skip when plotting, to avoid overcrowding the log

        for r in radii[::skip]:
            this_circle = plt.Circle(star, radius=r, edgecolor=aperture_color, facecolor='none')
            axes[0].add_patch(this_circle) # Plot apertures

    maxr = np.max(radii) # Largest radius, used to make a clipped array.

    clipped_array = image_array[star[1]-maxr:star[1]+maxr, star[0]-maxr:star[0]+maxr] # Small clipped array, used to compute distances from centre of apertures.

    clipped_origin = [maxr, maxr] # Centre of clipped array (by construction)

    distances = get_distances(clipped_array, origin=clipped_origin) # Get distance array

    counts = np.zeros_like(radii, dtype=np.float32) # Array to store star counts

    for rindex in range(len(radii)): # For each of the given radii

        r = radii[rindex]

        mask = distances < r  # Find all pixels that are distance r away
        area = np.sum(mask)   # Find number of these pixels

        # Compute the total counts, and subtract the background counts to get the counts due solely to the star.
        counts[rindex] = np.sum(mask*clipped_array) - bkg_counts_pp*area

    star_counts = np.mean(counts[-points_to_avg:]) # Average the last ``points_to_avg`` points.

    if(print_log):
        print(f"Star counts: {star_counts} (averaging over {points_to_avg} points).")

        # Plot the apertures
        axes[1].scatter(radii, counts, color='steelblue', ec='k', alpha=0.8)
        axes[1].axhline(star_counts, color='firebrick', ls='--', label=f"Final counts: {star_counts:.5}")
        axes[1].set_ylabel("Counts")
        axes[1].set_xlabel("Aperture radius (pixels)")
        axes[1].legend()

    return star_counts


def get_bkg_counts_pp(image_array, aperture_coord=None, aperture_rad = 10, n_apertures = 200, auto_reject=True, n_sigma=3.0, return_counts_and_areas=False, print_log=False, cmap="Greys_r", aperture_colors=['firebrick', 'k'], fig=None, axes=None):
    """
    Compute the average number number of background counts per pixel for a given image.

    A large number of apertures are chosen, and the number of counts in each of them is computed. Then, apertures are rejected based on their whether their numbers of counts are significantly far away from the mean value. These "outliers" are assumed to contain stars in them. The underlying assumption of this method is that the majority of the image makes up the background.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    aperture_coord: array_like or None, default: None
        2D array if ``(N,2)`` elements, each denoting the coordinates of the centre of an aperture of ``aperture_rad`` radius. By default, these coordinates are chosen randomly.

    aperture_rad: int or array_like, default: 20
        Radius in pixels of each aperture. If a single number if provided, all apertures have the same radius. If an array is provided, it should either be the same length as the ``aperture_coord`` (if it is provided) or ``n_points``, in that order.

    n_apertures: int, default: 200
        If ``aperture_coord`` is not provided, these are the number of apertures that will be initially sampled. This is ignored if ``aperture_coord`` is provided, and replaced by the length of ``aperture_coord``.

    auto_reject: bool, default: True
        Boolean flag that decides whether to automatically reject all data points until they all lie within ``n_sigma`` standard deviations away from the mean.

    n_sigma: float, default: 3.0
        Number of standard deviations away from the mean beyond which to reject all data points.

    return_counts_and_areas: bool, default: False
        Boolean flag to decide whether to return the number of counts and the areas of the apertures that were finally used (after automatically rejecting apertures, if chosen), in addition to the background counts per pixel. By default, this is not provided.

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it produces a figure with two plots: one which visually shows the star and all of the initially chosen apertures overlaid, and the other which shows the apertures that have been retained if ``auto_reject=True``.

    cmap: str, default: "Greys_r"
        A string containing the colormap title. Should be one of the colormaps used by matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html.

    aperture_colors: [str, str], default: ['firebrick', 'k']
        Array containing two elements, the first being the color of the unrejected apertures in the first plot, and the second being the color of the apertures after rejection in the second plot. Both colors must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    axes: ndarray of two elements which are matplotlib axes objects, default: None
        Axes on which to plot the result. By default, a new axis of two columns is created.

    Returns
    -------
    bkg_counts_pp: float
        The final averaged number of counts per pixel for all background pixels.

    counts, areas: array_like, optional: only returned if ``return_counts_and_areas=True``.
        Array of finally used counts and areas of each aperture that was used to compute ``bkg_counts_pp``.

    Raises
    ------
    ValueError
        If ``n_apertures`` or ``n_sigma`` is not positive.

    ValueError
        If ``aperture_rad`` is neither a number or a 1D array.

    ValueError
        If ``aperture_rad`` (or any element of ``aperture_rad``) is less than or equal to 1 pixel.

    ValueError
        If ``aperture_coord`` is not provided, and ``aperture_rad`` is neither a number, nor an array of ``n_apertures`` elements.

    ValueError
        If ``aperture_coord`` *is* provided, but it isn't a 2D array, or it doesn't have the same length as ``radii``.

    ValueError
        If multiple apertures and radii are provided, and not enough space is left between the largest possible aperture and the edge of the image.

    Usage
    -----
    >>> this_bkg_cpp = get_bkg_counts_pp(mizar_image, aperture_rad=10)
    """
    n_apertures = int(n_apertures) # Convert ``n_apertures`` to an integer

    if(n_apertures <= 0):
        raise ValueError("Number of background points must be at least one, but got: "+ str(n_apertures)+ " points.")

    if(n_sigma <= 0):
        raise ValueError("``n_sigma`` must be > 0, but got: "+ str(n_sigma))

    if(np.ndim(aperture_rad) == 0):
            radii = aperture_rad*np.ones(n_apertures)
    elif(np.ndim(aperture_rad)!= 1):
        raise ValueError("``aperture_rad`` must be a 1D array, but instead got an array of shape "+str(np.shape(aperture_rad)))
    else:
        radii = np.array(aperture_rad)

    radii = radii.astype(int)       # Convert radii to an arry of integers
    if(np.any(radii<=0)):           # Check if any of the radii are negative
            raise ValueError("Aperture radius must be at least one pixel, but got the following unacceptable values: "+ str(np.unique(radii[radii<=0]))+ " pixels.")

    Ly, Lx = np.shape(image_array)   # Get image array dimensions
    maxr = np.max(radii)             # Find largest radius in provided array:
                                     # this is used to find the distance arrays

    if(aperture_coord is None):      # If no aperture coordinates are given,
        # Randomly assign pixel values for coordinates
        points_x = np.random.randint(maxr, Lx - maxr, size=n_apertures)
        points_y = np.random.randint(maxr, Ly - maxr, size=n_apertures)

        if(len(radii) != n_apertures): # Check if ``radii`` has ``n_apertures`` elements
            raise ValueError("``aperture_rad`` should either be a number, or an array of ``n_apertures`` elements. ``n_apertures`` = "+str(n_apertures)+" but ``aperture_rad`` has "+str(len(radii)) + " elements.")

    else:                              # If aperture coordinates *are* given
        aperture_coord = np.array(aperture_coord) # Convert it to a NumPy array

        if(np.ndim(aperture_coord) != 2): # Check the dimensions of this array
            raise ValueError("``aperture_coord`` should be a two dimensional array of coordinates, but got "+str(aperture_coord)+".")

        if(np.ndim(aperture_rad)!= 0):
            if( len(aperture_coord)!= len(aperture_rad)):
                raise ValueError("``aperture_rad`` should either be a number, or an array with the same number of elements as ``aperture_coord``. Instead, their lengths are "+str(len(aperture_coord))+" and "+str(len(aperture_rad))+" respectively.")

        points_x, points_y = aperture_coord[:,0].astype(int), aperture_coord[:,1].astype(int)
        n_apertures = len(aperture_coord) # Reset ``n_apertures`` to be the number of coordinates provided

    if(print_log):
        if(fig is None or axes is None): # If no fig or axes are provided, create them
            fig, axes = plt.subplots(nrows=1, ncols=2, width_ratios=[1,1])

        axes[0].set_title("Before auto-rejection")
        axes[1].set_title("After auto-rejection")

        astrolab.imaging.display(image_array=image_array, cmap=cmap, fig=fig, ax=axes[0])

        for n in range(n_apertures): # Plot each aperture in the left plot
            this_origin = [points_x[n], points_y[n]]
            this_circle = plt.Circle(this_origin, radius=radii[n], edgecolor=aperture_colors[0], facecolor='none')
            axes[0].add_patch(this_circle)


    distances = get_distances(np.zeros((2*maxr, 2*maxr)), origin=[maxr,maxr]) # Create a tiny square array for distances

    counts = np.zeros(n_apertures, dtype=np.float32) # Empty array for counts
    areas  = np.zeros(n_apertures, dtype=np.float32) # Empty array for areas

    for n in range(n_apertures): # For each of the provided apertures
        background_box = image_array[points_y[n]-maxr:points_y[n]+maxr, points_x[n]-maxr:points_x[n]+maxr] # Create a small box whose centre is the centre of the aperture

        if(points_y[n]-maxr < 0 or points_y[n]+maxr > len(image_array) or points_x[n]-maxr < 0 or points_x[n]+maxr > len(image_array[0])):
            raise ValueError("Currently, coordinates cannot be placed too close to the edge if multiple radii are provided. In this case, since the largest circle is "+str(maxr)+" pixels, leave at least that much on either side for all points.")

        aperture_mask = distances < radii[n] # Create a mask to find all pixels less than this particular aperture's radius

        areas[n]  = np.sum(aperture_mask)     # Find the area (number of pixels)
        counts[n] = np.sum(aperture_mask*background_box) # Find the counts

    if(auto_reject):  # If we are automatically rejecting points,
        all_rejection_masks = [] # create an empty list to hold the rejection masks

        n_rem = n_apertures             # Initialise values
        old_num = n_apertures
        remaining_array = counts/areas  # Array of counts per pixel to start rejecting from

        while(n_rem>0):                 # Until *no* points are removed in successive iterations
            this_rejection_mask = reject_outliers(remaining_array, n_sigma=n_sigma)            # Create a rejection mask
            all_rejection_masks.append(this_rejection_mask) # append it to list

            new_num = np.sum(this_rejection_mask) # Store new number of elements
            n_rem   = old_num - new_num           # Find number removed
            old_num = new_num                     # Reset old number of elements

            remaining_array = remaining_array[this_rejection_mask] # Reset the array of elements and repeat the rejection process on it

        for m in all_rejection_masks: # Once all the rejection masks have been created
            points_x = points_x[m]    # Apply them one-by-one on the `x`
            points_y = points_y[m]    # and `y` coordinates,
            counts   = counts[m]      # the number of counts, and
            areas    = areas[m]       # the number of areas (i.e. no of pixels).

    bkg_counts_pp = np.round(np.mean(counts/areas), 4) # Find the background counts per pixel by dividing each of the counts by the appropriate areas

    if(print_log):
        astrolab.imaging.display(image_array=image_array, fig=fig, ax=axes[1])
        for n in range(len(points_y)): # Print the remaining apertures on the right plot
            this_origin = [points_x[n], points_y[n]]
            this_circle = plt.Circle(this_origin, radius=radii[n], edgecolor=aperture_colors[1], facecolor='none')
            axes[1].add_patch(this_circle)

        print("Background counts per pixel: ", bkg_counts_pp)

    if(return_counts_and_areas): # If we need to return the counts and areas
        return bkg_counts_pp, counts, areas # return them as well

    return bkg_counts_pp


def get_bright_star_catalog():
    """
    Load a bright-star catalogue of 1346 stars.

    Parameters
    ----------
    None

    Returns
    -------
    catalog: Pandas DataFrame
        A Pandas dataframe containing data of the 1346 bright stars and their data.

    Usage
    -----
    >>> bright_cat = get_bright_star_catalog()
    """
    try:
        catalog = pd.read_csv(StringIO(get_data(__name__, '/data/bright_star_catalog.csv').decode())) # Read package data for catalog
        return catalog
    except:
        raise RuntimeError("Could not find inbuilt bright-star catalog. This should not be happening and is a serious error. Please raise an issue on the GitHub page here: https://www.github.com/dpcherian/astrolab.")


def get_star_data(hr=None, name=None, catalog=None, hr_colname='HR', name_colname='name', requested_colnames=None, return_numpy=False):
    """
    Get a star's data from a catalog.

    Parameters
    ----------
    hr: int or None, default: None
        HR number of the star.

    name: str or None, default: None
        Common name of the star. Ignored if ``hr`` is provided.

    catalog: Pandas DataFrame or None, default: None
        Catalog of stars in which to search for requested star's data. By default, the 1346 bright-star catalog is used.

    hr_colname: str or None, default: "HR"
        Column name in catalog in which to search for HR number ``hr``.

    name_colname: str or None, default: "name"
        Column name in catalog in which to search for name ``name``.

    requested_colnames: str, list of str or None, default: None
        A column name or list of column names in the catalog whose values to return. By default, all columns are returned.

    return_numpy: bool, default: False
        Boolean flag to decide whether the function returns the data as a Pandas DataFrame, or a NumPy array.

    Returns
    -------
    star_data: Pandas DataFrame or NumPy array
        A Pandas dataframe or a NumPy array (if ``return_numpy=True``) containing the requested data of the specified star.

    Raises
    ------
    ValueError
        If neither ``hr`` nor ``name`` are provided.

    ValueError
        If star is not present in catalog.

    Warns
    -----
    UserWarning
        If both ``hr`` and ``name`` are provided, ``name`` is ignored.

    Usage
    -----
    >>> this_star_data = get_star_data(hr=5054)

    >>> this_star_data = get_star_data(name="mizar")
    """
    if(catalog is None): # If no catalog is provided, load the default catalog
        catalog = get_bright_star_catalog()

    if(hr is None and name is None): # If neither HR number nor name is provided
        raise ValueError("You must provide either the HR number or the name of the star!")                  # raise an error
    elif(hr is not None):            # If HR number is provided, it is used
        if(name is not None):        # If name is also provided, raise warning
            warnings.warn("Both HR number and name are given, ignoring ``name``.", UserWarning)

        star_data = catalog[catalog[hr_colname]==hr] # Get data from catalog
        error_string = f"HR number \"{hr}\""         # String for potential error message
    else:
        star_data = catalog[catalog[name_colname]==name]
        error_string = f"name \"{name}\""            # String for potential error message

    if(len(star_data) == 0): # If no results are found, raise appropriate error
        raise ValueError(f"The {error_string} is not present in the catalog. Check this value, or use an extended catalog.")

    if(return_numpy):
        return star_data[requested_colnames].to_numpy()[0]
    else:
        return star_data[requested_colnames]


def get_color_temperature_fits(catalog=None, deg=10, T_colname='Teff(K)', colors_colname=['STC_R', 'STC_G', 'STC_B'], return_fit=True, return_err=False, unpack=False, precision=0.01, print_log=False, colband_label=['R', 'G', 'B'], data_colors=['k', 'k', 'k'], fit_colors=['rebeccapurple', 'saddlebrown', 'dimgray'], fit_ls='solid', fit_alpha=0.33, fill_alpha=0.1, fig=None, ax=None):
    """
    Fit color and temperature data of stars from a catalog using a polynomial fit, and return the temperatures and fits. Optionally, the errors in the fits can also be returned.

    .. note:: Currently, the errors are equal for each point, and given by the standard deviation of all the points from the fit.

    Parameters
    ----------
    catalog: Pandas DataFrame or None, default: None
        Catalog of stars in which to search for requested star's data. By default, the 1346 bright-star catalog is used.

    deg: int, default: 10
        Order of polynomial fit.

    T_colname: str or None, default: "Teff(K)"
        Column name in catalog in which to search for temperature data.

    colors_colname: list of str, default: ['STC_R', 'STC_G', 'STC_B']
        A column name or list of column names where the magnitudes in different bands are stored in the catalog.

    return_fit: bool, default: True
        Optional flag to return the fit arrays. By default, these are returned.

    return_err: bool, default: False
        Optional flag to return the array of errors about each point. .. note:: Currently, these are constant values for each of the points corresponding to the standard deviation away from the fit, but point-wise errors could be added later.

    unpack: bool, default: False
        If True, the returned array is transposed, so that arguments may be unpacked using ``T, fitBmR, fitGmR, fitBmG = get_color_temperature_fits()``. By default, this is ``True``.

    precision: float, default: 0.01
        Minimum precision in the fitted temperature array. For example, is the temperature precise up to 12_000.0 K, or 12_000.01 K, or 12_000.001 K?

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it produces a plot with the fits and their standard deviation errors.

    colband_label: list of str, default: ['R', 'G', 'B']
        A list of labels used to denote the different bands in the plot.

    data_colors: list of color strings, default: ['k', 'k', 'k']
        Array of colours for the three sets of data-points that are to be fit. All colors must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    fit_colors: list of color strings, default: ['rebeccapurple', 'saddlebrown', 'dimgray']
        Array of colours for the three fits and their standard-deviation fills. All colors must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    fit_ls: str, default: 'solid'
        Linestyle of the fit lines. Linestyle should be one of the matplotlib "linestyles": https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html

    fit_alpha: 0 < float < 1, default: 0.33
        Alpha of the fit lines.

    fill_alpha: 0 < float < 1, default: 0.33
        Alpha of the standard-deviation fill around the fit lines.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    return_T: array_like
        Array of either the actual or an interpolated temperature data (based on whrther ``return_fit=True`` or not).

    return_colors: array_like
        Either an array of the actual data points (or the fitted data points, if ``return_fit=True``) for the colors.

    err_array: array_like (optional)
        Array of the errors about each data point. Returned only if ``return_err=True``.

    Usage
    -----
    >>> thisTemp, thisBmR, thisGmR, thisBmG = get_color_temperature_fits(unpack=True)

    >>> thisTemp, thisBmR, thisGmR, thisBmG, thisBmR_err, thisGmR_err, thisBmG_err = get_color_temperature_fits(unpack=True, return_err=True)
    """
    if(catalog is None): # If no catalog is provided, load the default catalog.
        catalog = get_bright_star_catalog()

    T = np.array(catalog[T_colname]) # Get temperature data from catalog
    sorted_T_indices = T.argsort()   # Sort this data
    T = T[sorted_T_indices]

    # Create an interpolating array of temperatures, with the given precision
    if(return_fit):
        n_points = int( (np.max(T) - np.min(T))/precision )
        fitT = np.linspace(np.min(T), np.max(T), n_points)

    color_array=np.zeros((len(catalog),3))  # Array of colour values to match
    color_array[:,0] = catalog[colors_colname[2]] - catalog[colors_colname[0]]
    color_array[:,1] = catalog[colors_colname[1]] - catalog[colors_colname[0]]
    color_array[:,2] = catalog[colors_colname[2]] - catalog[colors_colname[1]]

    labels = np.array([" ", " ", " "], dtype="U20") # Labels for plot
    labels[0] = colband_label[2] + " $-$ " + colband_label[0]
    labels[1] = colband_label[1] + " $-$ " + colband_label[0]
    labels[2] = colband_label[2] + " $-$ " + colband_label[1]

    polyfit_array = np.zeros_like(color_array) # Empty array to hold fits and
    err_array     = np.zeros_like(color_array) # error data

    if(return_fit):
        polyfit_array = np.zeros((len(fitT),3)) # Empty array to hold fits and
        err_array     = np.zeros((len(fitT),3)) # error data

    if(print_log):                       # Set up the plot
        if(fig is None or ax is None):
            fig, ax = plt.subplots()

        ax.set_ylabel("Color")           # Set the plot axes
        ax.set_xlabel("Temperature (K)") # and legend

    for i in range(3):                         # For each of the three colours
        y = color_array[:,i]
        y = y[sorted_T_indices]
        z = np.poly1d(np.polyfit(T,y,deg=deg))  # Fit a ``deg``-order polynomial
        std = np.std(y-z(T))

        if(print_log):
            # Plot the data and the fits, along with a one-sigma fill_between
            ax.scatter(T, y, s=1, color=data_colors[i], alpha=fit_alpha)
            ax.plot(T,z(T), color=fit_colors[i], ls=fit_ls, label=labels[i])
            ax.fill_between(T, z(T) - std, z(T) + std, color=fit_colors[i], alpha=fill_alpha)

        polyfit_array[:,i] = z(fitT) # Store the fit values
        err_array[:,i]   = std       # and the errors

    # Choose which array to return, based on ``return_fit``.
    return_colors = polyfit_array if(return_fit) else color_array
    return_T = fitT if(return_fit) else T

    # Choose whether to return the errors or not, based on ``return_err``.
    if return_err:
        return (return_T, *return_colors.T, *err_array.T) if (unpack) else (return_T, return_colors, err_array)
    else:
        return (return_T, *return_colors.T) if (unpack) else (return_T, return_colors)


def get_mags(target_counts, reference_counts, reference_hr=None, reference_name=None, catalog=None, hr_colname="HR", name_colname="name", colors_colname = ['STC_R', 'STC_G', 'STC_B']):
    """
    Get magnitude of a "target" star relative to a "reference" star whose absolute magnitudes are found in a catalog.

    Parameters
    ----------
    target_counts: array_like
        Array of photon counts in different filters for the target star

    reference_counts: array_like
        Array of photon counts in different filters for the reference star

    reference_hr: int or None, default: None
        HR number of the reference star.

    reference_name: str or None, default: None
        Common name of the reference star. Ignored if ``reference_hr`` is provided.

    catalog: Pandas DataFrame or None, default: None
        Catalog of stars in which to search for requested star's data. By default, the 1346 bright-star catalog is used.

    hr_colname: str or None, default: "HR"
        Column name in catalog in which to search for HR number ``hr``.

    name_colname: str or None, default: "name"
        Column name in catalog in which to search for name ``name``.

    colors_colname: list of str, default: ['STC_R', 'STC_G', 'STC_B']
        A column name or list of column names where the magnitudes in different bands are stored in the catalog.

    Returns
    -------
    target_mags: array_like
        An array of the target magnitudes.

    Raises
    ------
    AssertionError
        If the lengths of ``target_counts``, ``reference_counts``, and ``colors_colname`` aren't the same.

    Usage
    -----
    >>> alcor_mags = get_mags(alcor_counts, mizar_counts, reference_hr=5054)
    """
    target_counts = np.array(target_counts); reference_counts=np.array(reference_counts)

    assert len(target_counts) == len(reference_counts) == len(colors_colname), "``target_counts``, ``reference_counts``, and ``colors_colnames`` must all have the same length."

    # Fetch magnitude data for reference star from catalog
    reference_mags = get_star_data(hr=reference_hr, name=reference_name, catalog=catalog, hr_colname=hr_colname, name_colname=name_colname, requested_colnames=colors_colname, return_numpy=True)

    # Compute target magnitudes from reference magnitudes and counts
    target_mags = - 2.5*np.log10(target_counts/reference_counts) + reference_mags

    return target_mags


def get_temp(mags, catalog=None, T_colname='Teff(K)', colors_colname=['STC_R', 'STC_G', 'STC_B'], precision=0.01, print_log=False, colband_label=['R', 'G', 'B'], data_colors=['k', 'k', 'k'], fit_colors = ['rebeccapurple', 'saddlebrown', 'dimgray'], fit_ls='solid', fit_alpha=0.33, fill_alpha=0.1, fig=None, ax=None):
    """
    Get the temperature of a star given its magnitudes in different colour bands and a color-temperature fit obtained from a catalog.

    .. note:: Currently, the errors are equal for each point, and given by the standard deviation of all the points from the fit.

    Parameters
    ----------
    mags, array_like
        Array of magnitudes in different color bands.

    catalog: Pandas DataFrame or None, default: None
        Catalog of stars in which to search for requested star's data. By default, the 1346 bright-star catalog is used.

    T_colname: str or None, default: "Teff(K)"
        Column name in catalog in which to search for temperature data.

    colors_colname: list of str, default: ['STC_R', 'STC_G', 'STC_B']
        A column name or list of column names where the magnitudes in different bands are stored in the catalog.

    precision: float, default: 0.01
        Minimum precision in the fitted temperature array. For example, is the temperature precise up to 12_000.0 K, or 12_000.01 K, or 12_000.001 K?

    print_log: bool, default: False
        Provides option to print a log to debug your code. In this case, it produces a plot with the fits and their standard deviation errors, as well as lines indicating the corresponding temperatures for each color-band.

    colband_label: list of str, default: ['R', 'G', 'B']
        A list of labels used to denote the different bands in the plot.

    data_colors: list of color strings, default: ['k', 'k', 'k']
        Array of colours for the three sets of data-points that are to be fit. All colors must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    fit_colors: list of color strings, default: ['rebeccapurple', 'saddlebrown', 'dimgray']
        Array of colours for the three fits and their standard-deviation fills. All colors must be one of the matplotlib "named colors": https://matplotlib.org/stable/gallery/color/named_colors.html.

    fit_ls: str, default: 'solid'
        Linestyle of the fit lines. Linestyle should be one of the matplotlib "linestyles": https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html

    fit_alpha: 0 < float < 1, default: 0.33
        Alpha of the fit lines.

    fill_alpha: 0 < float < 1, default: 0.33
        Alpha of the standard-deviation fill around the fit lines.

    fig: matplotlib figure object, default: None
        Figure on which to plot the result. By default, a new figure is created.

    ax: matplotlib axes object, default: None
        Axes on which to plot the result. By default, a new axis is created.

    Returns
    -------
    T_eff: array_like
        Array temperatures obtained, one from each from the color-temperature calibration.

    Raises
    ------
    AssertionError:

    Warns
    -----
    UserWarning
        If the computed temperature does not lie within the temperature-range of the catalog data.

    Usage
    -----
    >>> mizar_temp = get_temp(mags=[2.0086,2.1897,1.1739], print_log=True)
    """
    assert len(mags) == len(colors_colname), "``mags`` and ``colors_colnames`` must have the same length."

    if(print_log and (fig is None or ax is None)):
        fig, ax = plt.subplots()

    temps, fit_BminusR, fit_GminusR, fit_BminusG = get_color_temperature_fits(return_fit=True, unpack=True, precision=precision, catalog=catalog, T_colname=T_colname, colors_colname=colors_colname, print_log=print_log, data_colors=data_colors, fit_colors = fit_colors, fit_ls=fit_ls, fit_alpha=fit_alpha, fill_alpha=fill_alpha, fig=fig, ax=ax) # Get fit-data.

    # Get colour data from magnitudes.
    col_BminusR = np.array(mags[2] - mags[0])
    col_GminusR = np.array(mags[1] - mags[0])
    col_BminusG = np.array(mags[2] - mags[1])

    # Lists to hold fits and colour data, so that we can loop over them.
    fits = [fit_BminusR, fit_GminusR, fit_BminusG]
    cols = [col_BminusR, col_GminusR, col_BminusG]

    T_eff = [] # Empty list to store the fitted T_eff data

    for i in range(len(fits)):
        # We will look for all points in the fit lower than the value we are trying to match, and pick out the corresponding temperature.

        matching_T_indices = np.argwhere(fits[i] < cols[i])[:,0] # np.argwhere takes in a boolean array finds all indices that are True, with each dimension in a different column. Since our fits[i] is 1D, we only need the zeroth column of this array.

        if(len(matching_T_indices)!=0): # If atleast one point satisfies condition,
            matching_T = temps[matching_T_indices[0]] # the first point is the answer

            if(len(matching_T_indices)==len(fits[i])): # If *all* points satisfy it, raise a warning
                warnings.warn("Temperature below the range of available temperature data. Be careful when you interpret your results.", UserWarning)

        else: # If no points satsify the condition, use the temperature of the lowest fits value and raise a warning
            min_of_fit = np.argmin(fits[i]) # Index of lowest value
            matching_T = temps[min_of_fit]  # Associated temperature

            warnings.warn("Temperature above the range of available temperature data. Be careful when you interpret your results.", UserWarning)

        T_eff.append( matching_T )

    if(print_log): # If a log is to be printed, also plot lines marking computed temperatures
        handles, labels = ax.get_legend_handles_labels() # Get the old legend

        for i in range(len(cols)):
            labels[i] = f"{labels[i]} : {T_eff[i]:.0f}" # Add the temperature to the labels
            ax.axhline(cols[i], color=fit_colors[i], ls='--')
            ax.axvline(T_eff[i], color=fit_colors[i], ls='dashdot')
        ax.legend(labels=labels, handles=handles)

    return np.round(T_eff,4)


def get_distances(image_array, origin):
    """
    Create an array whose elements contain the pixel distances from a provided origin. This is used to mask out and count pixel values within a circular region.

    Parameters
    ----------
    image_array: array_like
        A 2D array which serves as the image data.

    origin: [int, int]
        A 1D array containing the pixel location of the point from which distances are measured.

    Returns
    -------
    distances: array_like
        A 2D array containing the distances.

    Raises
    ------
    ValueError
        If ``origin`` is not a 1D array with two elements.

    Warns
    -----
    UserWarning
        If ``origin`` does not lie within the image array.

    Usage
    -----
    >>> dist_array = get_distances(this_image, origin=[512,512])
    """

    if(np.array(origin).ndim != 1 or len(origin)!=2): # If origin isn't a 1D array with two elements, raise an error
        raise ValueError("``origin``: expected a 1-dimensional array with two elements, but got \""+str(origin)+"\"")

    if(origin[0] < 0 or origin[0] > len(image_array[0]) or origin[1] < 0 or origin[1] > len(image_array)):
        warnings.warn("``origin`` doesn't lie within the image-array. I hope you know what you're doing!")

    distances = np.zeros_like(image_array) # Create a 2D array to store distances

    for i in range(0, len(image_array)):
        for j in range(0, len(image_array[0])):
            distances[i, j] = np.sqrt((i - origin[0])**2 + (j - origin[1])**2) # Compute distance from ``origin`` to every other pixel
    return distances


def reject_outliers(array, n_sigma=3.0):
    """
    Reject all elements of an array that are greater than some number of standard deviations away from the mean, and return a mask of the rejected elements.

    Parameters
    ----------
    array: array_like
        A 1D array of float data.

    n_sigma: float, default: 3.0
        Number of standard deviations away from the mean beyond which to reject data points.

    Returns
    -------
    mask: boolean 1D array
        A boolean mask of the same length as ``array``, indicating which data points must be rejected.

    Usage
    -----
    >>> this_mask = reject_outliers([1,2,8,10,23,4,2,0.1,3], n_sigma=0.5)
    """
    array = np.array(array, dtype='float64') # Convert the array to an array of floats

    mean = np.mean(array) # Find the mean of the array
    std  = np.std(array)  # Find the standard deviation of the array

    max_val = mean + n_sigma*std # Maximum value beyond which to reject data
    min_val = mean - n_sigma*std # Minimum value below which to reject data

    mask = (array >= min_val) & (array <= max_val) # Creating the mask

    return mask
