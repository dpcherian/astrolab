Documentation for the python package `astrolab`
===============================================

This GitHub repository contains documentation for the python package `astrolab` developed for _Observering the Cosmos_, the introductory astronomy laboratory at Ashoka University. This documentation is hosted on ReadTheDocs [here](https://ashoka-astrolab.readthedocs.io/).

### Prerequisites

- Python 3 - Sphinx is based on Python, therefore you must have Python installed
- A Python package installer like `pip` or `conda`

### Setting up the documentation environment

The following commands will make a local copy of this repository, and install the required python packages.

- Clone the repository: `git clone git@github.com:dpcherian/phy1080-astrolab.git`
- Move into the repository: `cd phy1080-astrolab`
- Install requirements: `pip install -r requirements.txt` or `conda install --file requirements.txt`

### Building the Sphinx site locally

You can preview the final version of your local edits by building the HTML pages from your source. Note that this is only to check the final documentation. **The build files should not be committed to version control.** The final documentation site is built automatically from source files.

In order to build a local version of the HTML pages, navigate to within the repository and:

- If you have `make` installed, simply run `make html`
- If not, run `sphinx-build -b html source build/html`

You can now edit the source files and preview the results using any text editor. Sphinx uses [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html), a "a simple, unobtrusive markup language" that will not take you too long to learn if you have not already used it. 

If you wish, you can use a code editor like [VS Code](https://code.visualstudio.com) to edit the files, along with the [reStructuredText plugin](https://docs.restructuredtext.net) which includes many additional features, including a live preview.

## License

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]
