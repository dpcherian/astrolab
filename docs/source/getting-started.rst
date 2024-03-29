Introduction
============

In this section, we will cover setting up Python on your machine (using the "Miniconda" Python distribution), creating a virtual environment for this course, and installing the ``astrolab`` package in that environment. Follow these instructions carefully.

.. note:: If you find any errors with this part of the documentation, please contact the maintainers of this package as soon as possible.


Setting up Python
=================

.. caution:: This is still a work in progress, and the installation instructions have not been tested over a range of operating systems, and therefore it is conceivable that it cause errors with your Python installation. As always, it is **strongly advised** that you use a conda environment for this lab and for this package. That way, if anything goes wrong, your Python installation's integrity is not compromised.


Getting and installing ``conda`` on your machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Those of you who have already used Python before might be familiar with the `Anaconda Python distribution <https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)>`_. Anaconda comes with its own "package manager", ``conda``, which automates the process of installing, upgrading, configuring, and removing Python packages. This is particularly useful if a package you need has multiple dependencies, or if you require multiple Python versions for different purposes on the same computer.

Anaconda is very useful, but is terrible bloated. It comes with a huge number of packages, most of which you will never use, either in this lab or elsewhere. Consequently, it sometimes slows down your computer. To avoid this, it is strongly recommended that you use **miniconda**. Miniconda is a minimal installer for the ``conda`` package manager, which comes with a couple of basic packages. The advantage of this is that you can only install what you need, keeping your Python installation lean and robust. The downside is that you need to install what you need at least once.

Begin by downloading miniconda from `here <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_, following the instructions appropriate to your operating system. Use all the default settings unless you know what you are doing. Once you have installed you should be able to use the ``conda`` command from your terminal.

.. caution:: Windows users will have to open the `Anaconda Command Prompt` that can be found in the start menu. Macintosh and Linux users can use the default system terminal.

If you follow the default settings, you should see the word ``(base)`` before your terminal prompt, as shown below. This is to show that the system recognises ``conda`` and its commands, and that you are using the "base" environment. We will come to environments in the next section: they are an easy way to sandbox the different packages you need for different purposes, so that they do not interfere with each other.

.. code-block:: bash

	(base) username@system ~$


You can now check the list of installed packages by running 

.. code-block:: bash

	conda list
	
This should show you the list of packages installed in your system. This should be a relatively short list. You will now need to add to it.

The command ``conda install <list of packages>`` will install packages in your base environment. If this is your first time using Python, it is advisable that you run the following command to install some important package:

.. code-block:: bash

	conda install jupyter numpy matplotlib scipy pandas mamba
	
The above command installs

- Jupyter Notebook: An interactive notebook in which you will be writing your Python code. In these interactive Python Notebooks (or "IPython Notebooks", which explains their filename extension "ipynb"), you can execute single lines of code or blocks of code in single cells. This can be very useful as you create a larger script.

- NumPy: A Python library which adds support for multi-dimensional arrays and matrices, along with a large collection of mathematical functions to operate on these arrays. Numpy arrays have the advantage of behaving in mathematical operations exactly as vectors do in physics.

- Matplotlib: The definitive plotting library for Python and NumPy. It is terribly versatile, and can -- and should! -- be used to create a variety of different kinds of plots.

- SciPy: A useful Python package that contains a large number of modules for mathematical operations common in science and engineering (for example, optimization, linear algebra, integration, interpolation, special functions, fourier transforms, signal and image processing, solving ordinary differential equations, and so on).

- Pandas: Another very useful Python package to deal with large amounts of data. We will not be using it extensively in this lab, but it will be helpful when we work with large datasets that can contain data of different datatypes (like our star catalogs).

- Mamba: Mamba is a replacement for the already described ``conda`` command. In everything that follows from here on, you can use ``mamba`` in the place of ``conda``. The reason for using it is that it is tremendously faster than ``conda`` in many cases. However, **be warned**, ``mamba`` must `only` be installed in the ``base`` environment. Installing it in other environments is `not supported <https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html>`_. However, once installed, it can then be used from every other environment.


Setting up a ``conda`` environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that you have set up your ``base`` environment, let us move on to setting up an environment for this course. A Python environment is a directory that contains a set of Python packages. You may define multiple such environments, and ``conda`` will make sure that the packages in one environment do not interfere with those in another environment. Thus, changes in one environment do not affect any of your other environments, and -- if things go wrong -- it's not necessary to reinstall Python from scratch on your system, merely the packages in a single environment.

You can create a new environment using the ``conda create`` (or ``mamba create``) command

.. code-block:: bash

	conda create --name ast1080 python=3.10
	
This command tells ``conda`` to create a new environment with the name ``ast1080``, and with Python 3.10 installed in it.

.. note:: 
	As with most ``bash`` commands, you don't always have to use the full form ``--name``, but can instead use the short-hand ``-n`` which would have the same effect. In other words, the command below has exactly the same effect as the one above: 
	
	.. code-block:: bash
	
		conda create -n ast1080 python=3.10

You can now get a list of the environments you have on your system by running 

.. code-block:: bash

	conda info --envs
	
You should see something like this:

.. code-block:: bash

	# conda environments:
	#
	base                  *  <your_root_directory>/miniconda3
	ast1080                  <your_root_directory>/envs/ast1080


Note that you now have two environments, one called ``base`` and one called ``ast1080``. The star ("*") above indicates which environment you are currently using, as does the word in parentheses before your prompt. Both of these things should tell you that you are currently working in the ``base`` environment. You should now shift to the ``ast1080`` environment. This can be done using the command

.. code-block:: bash

	conda activate ast1080


Now, install the important packages for this laboratory. You could install the same packages as above. In addition, you should also install ``astropy``, a collection of Python packages designed explicitly for use in astronomy. The command below installs all required packages

.. code-block:: bash

	conda install jupyter numpy matplotlib scipy pandas astropy

.. note:: You will notice that while we have installed most of the same packages as you did in the ``base`` environment, we have not installed ``mamba``. This is because, as we mentioned before, installing the ``mamba`` package on any environment other than the ``base`` environment is `not supported <https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html>`_.

Congratulations! You now have a working ``conda`` environment on your machine!


Setting up ``astrolab``
=======================


Installation
~~~~~~~~~~~~

The Python package written for this lab can be installed as follows:

.. warning:: It is strongly recommended that you use a new ``conda`` environment unless you know what you're doing.

- Create a new ``conda`` environment as described above: ``conda create -n ast1080 python=3.10``
- Activate this environment: ``conda activate ast1080``
- Install ``astrolab``: ``pip install git+https://github.com/dpcherian/astrolab``

Installing the ``astrolab`` package will automatically install all the packages it depends on (its "dependencies"). If you have followed the instructions in the section above, all of them should already be installed. Additionally, you should install Jupyter Notebook in the environment if you plan on using it to run your Python codes. As before, this can be done using 

.. code-block:: bash

	conda install jupyter

Once this is done, you could run the following Python code (either in the terminal or in a new Jupyter Notebook) which will test if the ``astrolab`` package has been installed, and print the currently installed version:

.. code-block:: python

	import astrolab
	print(astrolab.__version__)


Basic Usage
~~~~~~~~~~~

Once you have successfully installed the above package, you should be able to call the modules and use the functions they provide in any of your Jupyter Notebooks. For example, you could say

.. code-block:: python

	from astrolab import imaging as im
	

and you could then use the range of functions provided in the ``imaging`` library of ``astrolab`` to perform basic image-reduction and analysis. These functions are described in more detail in :ref:`the section describing the modules <astrolab>`.

Similarly, you could run:

.. code-block:: python

	from astrolab import timing as time

and use the range of functions of the ``timing`` library of the ``astrolab`` package to analyse ``.wav`` files for the Doppler Effect experiment. This, too, has been detailed in :ref:`the section describing the modules <astrolab>`.
