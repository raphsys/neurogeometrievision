# GEMINI.md - NeuroGeomVision

This file provides instructional context for working on the NeuroGeomVision project with Gemini.

## Project Overview

NeuroGeomVision is a computational vision library based on the neurogeometry of the visual cortex and Spiking Neural Networks (SNNs). It implements mathematical and neurophysiological models from Jean Petitot's "Neurogeometry of Vision". The goal is to create low-level vision algorithms that are neurophysiologically plausible, mathematically elegant, and computationally efficient.

The project is structured into several modules:

-   `neurogeomvision/core`: Basic utilities.
-   `neurogeomvision/retina_lgn`: Retina/LGN filters and spike encoding.
-   `neurogeomvision/v1_simple_cells`: Simple cells of the V1 cortex.
-   `neurogeomvision/contact_structure`: Contact space and jet space geometry.
-   `neurogeomvision/sub_riemannian`: Contour integration by geodesics.
-   `neurogeomvision/association_field`: Horizontal connections of V1.
-   `neurogeomvision/snn`: Spiking Neural Network implementations.
-   `neurogeomvision/learning_plasticity`: Synaptic plasticity and learning.

## Dependencies

The project's dependencies are listed in the `requirements.txt` file:

-   `numpy>=1.21.0`
-   `scipy>=1.7.0`
-   `torch>=1.9.0`
-   `matplotlib>=3.4.0`
-   `pillow>=8.3.0`
-   `opencv-python>=4.5.0` (Optional for advanced image processing)
-   `jupyter>=1.0.0` (Optional for notebooks)
-   `scikit-image>=0.18.0` (Optional for image utilities)

## Building and Running

### Installation

1.  Clone the repository.
2.  Create a virtual environment: `python -m venv .ngv-venv`
3.  Activate the virtual environment: `source .ngv-venv/bin/activate`
4.  Install the dependencies: `pip install -r requirements.txt`

### Running Tests

The project contains a number of test files in the `examples/` directory. These can be run as Python scripts. For example:

```bash
python examples/test_retina_v1.py
```

## Development Conventions

The project follows a modular structure, with each module containing its own README file with detailed documentation. The code is written in Python and uses PyTorch for tensor operations. The project aims to be extensible and compatible with neuromorphic hardware.

## Directory Overview

-   `.ngv-venv/`: The project's virtual environment.
-   `examples/`: Example scripts and tests for the different modules.
-   `neurogeomvision/`: The main source code for the library.
-   `tests/`: An empty directory, which will likely contain the project's test suite.

## Key Files

-   `requirements.txt`: The project's dependencies.
-   `setup.py`: The project's setup script (currently empty).
-   `neurogeomvision/README.md`: The main README file for the project, containing detailed documentation.
-   `neurogeomvision/retina_lgn/filters.py`: Implements the basic filters that simulate early visual processing in the retina and LGN.
-   `neurogeomvision/retina_lgn/coding.py`: Implements different neural coding strategies that transform analog intensities into discrete spike trains.
-   `neurogeomvision/v1_simple_cells/gabor_filters.py`: Implements Gabor filters to model the simple neurons of the primary visual area V1.
