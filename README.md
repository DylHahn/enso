# Western Tropical Pacific Subsurface Predictors of ENSO

This repository contains the code and supporting files for the paper **“Subsurface Western Pacific Temperature Precursors of Eastern Pacific ENSO Surface Variability.”**

The project tests whether subsurface temperature anomalies in the western tropical Pacific provide useful precursor information for future eastern Pacific ENSO surface conditions. The analysis uses monthly GODAS potential temperature data, computes temperature anomalies at multiple depth levels, applies ordinary least squares regression at lead times from 1 to 18 months, and includes an event-based analysis of western Pacific subsurface cooling before major El Niño events.

## Repository contents

```text
enso/
├── README.md
├── environment.yml
├── enso.py
├── west_east_event.py
├── enso_bootstrap.py
├── enso_jackknife.py
├── correlation.ipynb
├── data/
├── figures/
└── stats/
```

## Main files

* `enso.py`: Main analysis script for regression experiments, prediction skill calculations, and manuscript figures.
* `west_east_event.py`: Event-based western-to-eastern Pacific precursor analysis.
* `enso_bootstrap.py`: Bootstrap uncertainty analysis.
* `enso_jackknife.py`: Jackknife sensitivity analysis.
* `environment.yml`: Conda environment file for reproducing the analysis.

## Data files

The raw GODAS NetCDF files are **not included** in this GitHub repository because they are large.

Small processed files, such as moving-average anomaly text files, may be included in the `data/` folder when needed for the analysis. Larger raw or processed data files should be stored separately in the project’s Zenodo archive.

Expected local data files include GODAS-derived temperature anomaly and time-series files used by the analysis scripts.

## Figures and statistical outputs

The `figures/` folder contains generated manuscript figures.

The `stats/` folder contains saved statistical outputs from the regression, bootstrap, and jackknife analyses, including correlation and RMSE results used to create prediction skill summaries.

Large output files that are too large for GitHub should be archived with the Zenodo release instead.

## Reproducing the analysis

First, create the conda environment:

```bash
conda env create -f environment.yml
conda activate enso-paper
```

Then run the main analysis:

```bash
python enso.py
```

Run the event-based precursor analysis:

```bash
python west_east_event.py
```

Optional uncertainty and sensitivity analyses can be run with:

```bash
python enso_bootstrap.py
python enso_jackknife.py
```

Generated figures and statistical outputs will be saved locally to the `figures/` and `stats/` folders.

## Data availability

The ocean temperature data used in this project come from the NCEP Global Ocean Data Assimilation System (GODAS).

Raw GODAS NetCDF files are excluded from this GitHub repository. Processed GODAS-derived files and any large statistical outputs needed to reproduce the manuscript will be archived with the Zenodo release.

Zenodo DOI: TBA

## Citation

If you use this code, processed data, statistical outputs, or figures, please cite the archived Zenodo release and the associated manuscript.
