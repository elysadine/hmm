# HMM Analysis of Subtraction Error Trajectories in Madagascar

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the data, code, and documentation for the paper:

> **"Modeling the evolution of subtraction schemes in low-resource contexts: A hidden Markov analysis of persistent errors and learning trajectories"**  
> *International Journal of Educational Research* (JIJER-D-26-03234)  
> Razafindrafara, E.M.A., Razafinirina, M.A., Nguala, J.B., & Raherinirina, A.

## Overview

This repository provides the complete materials to reproduce the Hidden Markov Model (HMM) analysis of 845 students' subtraction performance data from a Teaching at the Right Level (TaRL) remediation program in Madagascar.

### Key Findings

- Identification of five latent procedural states (C0–C4)
- High persistence of the "Smaller-from-Larger" inversion error state (C2)
- Estimated time to mastery: 22.4 hours from C2 vs. 5.6 hours from C3

## Repository Structure
hmm/
├── README.md # This file
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
├── hmm.xlsx # Main dataset (845 students)
├── hmmsubstraction.xlsx # Subtraction-specific data
├── codehmm.py # Main HMM estimation script
├── codehmm2.py # Additional analysis scripts


## Data Description

### `hmm.xlsx`

Main dataset containing 845 students with the following variables:

| Variable | Description |
|----------|-------------|
| `ID` | Student identifier |
| `EPP` | School name |
| `CLASSE` | Grade level (2–5) |
| `Nombre d'heures` | Total remediation hours |
| `AGE` | Student age |
| `GENRE` | Gender (1 = male, 2 = female) |
| `Zone` | Geographic zone (1–3) |
| `PRETEST*` | Pre-test scores (addition, subtraction, multiplication, division) |
| `TESTFINAL*` | Post-test scores (addition, subtraction, multiplication, division) |

### `hmmsubstraction.xlsx`

Subtraction-specific dataset with procedural error coding:

- **Observable procedures (o1–o5):**
  - `o1`: Canonical correct procedure
  - `o2`: Smaller-from-Larger inversion
  - `o3`: Forgotten regrouping
  - `o4`: Incorrect decomposition
  - `o5`: Other/mixed procedure

- **Latent states (C0–C4):**
  - `C0`: Naive entry
  - `C1`: Mechanical execution
  - `C2`: Stabilized obstacle
  - `C3`: Partial understanding
  - `C4`: Expert mastery

## Requirements

## Python Dependencies

### Required Packages

This project requires the following Python packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24.0 | Numerical computations |
| `pandas` | >= 2.0.0 | Data manipulation and analysis |
| `scipy` | >= 1.10.0 | Scientific computing |
| `matplotlib` | >= 3.7.0 | Plotting and visualization |
| `seaborn` | >= 0.12.0 | Statistical data visualization |
| `openpyxl` | >= 3.1.0 | Excel file reading/writing |
| `hmmlearn` | >= 0.3.0 | Hidden Markov Model estimation |
| `statsmodels` | >= 0.14.0 | Statistical models and tests |
| `jupyter` | >= 1.0.0 | Interactive notebooks |

### Installation

To install all dependencies, run:

```bash
pip install -r requirements.txt
