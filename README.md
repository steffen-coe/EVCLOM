# traganalysis

## About

traganalysis is an analysis framework for atmospheric trace gas concentration measurements written in Python. It is available at [https://gitlab.com/SteffenCoe/traganalysis](https://gitlab.com/SteffenCoe/traganalysis) and can be downloaded or cloned from there freely. It is available under an [MIT License](LICENSE).

## Usage

The software framework features a modular structure, built to be reusable, easily readable, and powerful. The author hopes that future studies on atmospheric trace gas measurements can make use of this software package, facilitating and simplifying a certain set of study applications via pre-built and scalable methods.

It was written in the Python programming language and used on the Python version 3.8 (downward capability given at least to Python 3.5).

![Overview of the software framework](img/framework_overview.png)

### The `Dataset` class

The main constituent within the framework is the `Dataset` class. It serves as a container class on which all featured operations can be called. This comprises reading in, preprocessing, and cleaning the given concentration data and meteorological measurements, as well as calculating basic descriptive statistics, utilizing plotting tools, and conducting receptor modeling-specific studies such as principal component analysis and non-negative matrix factorization. The concentration data is stored in the class variable `self.df`, which leverages the powerful `pandas.DataFrame` object type. This enables robust indexing and labeling for quick access to the data.

### The `NMF` class

Furthermore, a designated class `NMF` encloses information around one specific non-negative matrix factorization that is conducted as part of a given study. It stores information about the included gases, their uncertainties, the initial concentration data, the two matrices that the input data was decomposed into ($X \approx WH$), as well as metadata on the NMF's calculational method such as the number of factors or iterations that were needed to find the solution. Using this class, plots of NMF source factor contributions and time series can be created, and the $Q$-value can be studied. Using key parameters, the framework assigns each instance of this class a unique label with which the respective NMF can be identified and reused. 

### The `Station` class

An additional class named `Station` loads and stores useful information on the receptor station at which the respective measurements were recorded, such as the site's geographic location and altitude, the GAW-ID, or other helpful identifiers that can be customized. 

### Configuration (`config.cfg` and `GLOBAL.py`)

The user can provide custom configuration files to be read by the framework. These files specify study-specific information such as the set of measured gas species, groups of these gases to be studied as subsets, or the start and end dates of the measurement series. This is done via the main configuration file `config.cfg`. In addition, the file `GLOBAL.py` contains a number of variable definitions that are used throughout the package.

### The `methods.py` file

The file `methods.py` contains helpful functions that are specific to the application of dealing with, for example, wind directions and objective weather types. For instance, the function `get_wind_dir()` provides wind direction sectors in a desired level of accuracy (for instance, W-NW-N vs. W-WNW-NW-NNW-N). In addition, `utils.py` provides useful plotting and other tools that are used by some functions in both the `Dataset` and `NMF` classes. All functions were developed by the author.

## Examples

All IPython notebooks (`.ipynb`) in the repository contain examples of how to use this software. For an example that applys practically all member functions of the `Dataset` class, see `plot.py`.

## Licensing

See the [LICENSE](LICENSE) file for licensing information as it pertains to
files in this repository.

## Requirements

Please see the [requirements](requirements.txt) for Python requirements and dependencies.

## Contact

The author is happy to be contacted via his [LinkedIn account](https://www.linkedin.com/in/steffen-coenen/) or via email (steffen.coenen [at] rwth-aachen.de) for any questions or help regarding the use of this software framework.

## Documentation

Further documentation can be found in the code. Each function contains its own documentation with description of all function parameters.
