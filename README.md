

<div align="center">
  <img src="static/images/Drift_Lens_Logo_cropped.png" width="300"/>
  <h4>DriftLens Demo: Real-time Unsupervised Concept Drift Detection for Deep Learning Models</h4>
</div>
<br/>


*DriftLens* is an **unsupervised drift detection** framework for deep learning classifiers on unstructured data.
It performs distribution distances between a historical dataset, called baseline, and a new data stream divided in fixed-sized windows.

# Table of Contents

- [Methodology and Models](#methodology-and-models)
- [DriftLens Demo](#demo)
  - [Screenshots](#screenshots)
- [Setup](#setup)
- [References](#references)
- [People involved](#people-behind-driftlens)

# Methodology and Models
![Screenshot](static/images/driftlens-demo-architecture.png)
The Drift Detection methodology in DriftLens includes two main phases: an *offline* and an *online*phases.

## Offline Phase
The *offline* phase takes as input an historical dataset, called baseline, that represents what the model learned during training. Firstly, the baseline data is feed into the model to extract the embedding vectors and the predicted labels 
&#x2460;. Then, the majority of the baseline dataset is used to model the distributions of the baseline &#x2461;. Instead, a small portion of the baseline data is used to estimate the threshold values &#x2462;
## Online Phase

# Setup
To use the tool locally:
1) Create and Start a new environment:
```sh
conda create -n driftlens-demo-env python=3.8 anaconda
conda activate driftlens-demo-env
```
2) Install the required packages:
```sh
pip install -r requirements.txt
```
3) Download the pre-computed embedding in the tool (TODO):
```sh
./download_data.sh 
```
4) Start the DriftLens app locally:
```py
python driftlens_app.py
```
The DriftLens app will run on localhost: http://127.0.0.1:5000

# References
```bibtex
@INPROCEEDINGS{driftlens,
  author={Greco, Salvatore and Cerquitelli, Tania},
  booktitle={2021 International Conference on Data Mining Workshops (ICDMW)}, 
  title={Drift Lens: Real-time unsupervised Concept Drift detection by evaluating per-label embedding distributions}, 
  year={2021},
  volume={},
  number={},
  pages={341-349},
  doi={10.1109/ICDMW53433.2021.00049}
  }
```

# People behind DriftLens

- **Salvatore Greco** - [Homepage](https://grecosalvatore.github.io/) - [GitHub](https://github.com/grecosalvatore) - [Twitter](https://twitter.com/_salvatoregreco)
- **Bartolomeo Vacchetti** - [Homepage]()
- **Daniele Apiletti** - [Homepage]()
- **Tania Cerquitelli** - [Homepage]()