(front-setup)=
# Software setup

The notebooks need Python 3.11, 3.12 or 3.13 with NumPy, SciPy and Matplotlib.
Building the website and the PDF notes needs a few more packages, all listed
in the repository's `requirements.txt` and `environment.yml`.

## Running a notebook without installing anything

Every notebook page of the website has a rocket icon at the top. It opens the
notebook on [Binder](https://mybinder.org) or
[Google Colab](https://colab.research.google.com). Notebooks that use a
dataset download it automatically from its public source when they do not
find the local copy.

## Installing locally with pip

```text
git clone https://github.com/aalsammani/Scientific-Computations-I-Notebooks.git
cd Scientific-Computations-I-Notebooks
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

## Installing locally with conda

```text
conda env create -f environment.yml
conda activate nla-course
jupyter lab
```

The notebooks are in the `book/` folder, one folder per chapter. The lab
notebook of Chapter 1, for example, is the file
`book/ch01-matrix-computations/lab.ipynb`. Each notebook is self-contained and
should be run from top to bottom in a fresh kernel.

## Building the website and the PDF notes

From the repository root:

```text
jupyter-book build book            # website in book/_build/html
python scripts/build_pdf.py        # PDFs in pdf/ (needs a LaTeX installation)
```

The website build executes every notebook, so it takes several minutes. The
PDF build needs `latexmk` and `pdflatex`, which are part of TeX Live and
MiKTeX.

## Checking your installation

```text
python -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)"
python -m pytest -q
```

The test suite executes all notebooks and checks the numerical claims made in
the notes.
