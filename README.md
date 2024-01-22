| ![image](https://github.com/dpcherian/astrolab/assets/19914486/72b45b24-5280-4d8a-b185-de3490dac2a2) | Welcome to ``astrolab`` on GitHub |
|----|--------------------------------------------------------|

The ``astrolab`` package was developed to accompany the introductory astronomy laboratory at Ashoka University. This laboratory, entitled "Observing the Cosmos", is aimed at introducing first-year undergraduates to the basics of observational astronomy, using data that they collect themselves as far as possible. To this end, the laboratory includes experiments related to timing, image reduction, spectroscopy, and photometry, in addition to more basic experiments that introduce students to coordinate systems and using astronomical equipment like telescopes and CMOS cameras.

Most of these experiments rely heavily on programming, a skill that many of the undergraduates who take this course have little to no prior knowledge of. As a compromise, this Python package was developed over the course of the year 2022-2023, both during and immediately after the first iteration of the laboratory. The idea of this package is to abstract out many of the complicated details that deal with how certain actions can be implemented programmatically, so that students can instead focus on exactly how their data should be processed, without unnecessary distractions.

The package contains two folders:

- **[astrolab](./astrolab)** which contains the source code for the ``astrolab`` package, and
- **[docs](./docs)** which contains the source code for the documentation written in Sphinx and hosted [here](https://astrolab.readthedocs.io).


## Installing ``astrolab``

> ⚠️ It is strongly recommended that you create a new ``conda`` environment for this package, to avoid conflicts with pre-existing Python packages in your system.

- Create a new ``conda`` environment: ``conda create -n ast1080 python=3.10``
- Activate this environment: ``conda activate ast1080``
- Install ``astrolab`` from this repository: ``pip install git+https://github.com/dpcherian/astrolab.git``

Once this is done, you can open Python and ``import`` the different modules in the ``astropy`` package.

Additionally, you can install Jupyter Notebook, which provides a convenient environment to work in. This can be done by running the ``conda install jupyter`` command.


## Contributing to ``astrolab``

#### **Do you have questions about this package?**

* If you have any doubts about how to use ``astrolab``, first go through the documentation on the [website](https://astrolab.readthedocs.io).
* If you need more assistance, write to us! We're always happy to help.


#### **Do you think you've found a bug in ``astrolab``?**

* **Ensure that this bug was not already reported** by searching on GitHub in [astrolab issues](https://github.com/dpcherian/astrolab/issues).
* If you're unable to find an open issue addressing your problem, open a new one in the corresponding repository. Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or a **test case** demonstrating the expected behavior that is not occurring.


#### **Have you managed to fix a bug or add a new feature?**
- Open a new GitHub [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) if you think your addition is worth being included in the relevant repository. This is applicable to contributions towards the actual code of the package, as well as those that concern the documentation.
- Ensure that the pull request description clearly describes the problem and solution or the new addition. Include the relevant issue number if applicable.


## License

[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

This work is licensed under the [MIT License](./LICENSE).
