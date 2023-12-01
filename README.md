

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
- [Recorded Demo](#recorded-demo)
- [References](#references)
- [People Involved](#people-behind-driftlens)

# Methodology and Models
![Screenshot](static/images/driftlens-demo-architecture.png)
The Drift Detection methodology in DriftLens includes two main phases: an *offline* and an *online* phases.

<table>
  <caption>Pre-uploaded use cases.</caption>
  <thead>
    <tr>
      <th>Use Case</th>
      <th rowspan="3">Dataset</th>
      <th rowspan="3">Domain</th>
      <th>Models</th>
      <th>F1</th>
      <th rowspan="3">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1.1</th>
      <td rowspan="3">Ag News</td>
      <td rowspan="3">Text</td>
      <td>BERT</td>
      <td>0.98</td>
      <td rowspan="3"> <b>Task</b>: Topic Classification. <BR>
<b>Training Labels</b>: <i>World</i>, <i>Business</i>, and <i>Sport</i> <BR>
<b>Drift</b>: Simulated with one new class label: <i>Science/Tech</i></td>
    </tr>
    <tr>
      <th>1.2</th>
      <td>DistillBERT</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>1.3</th>
      <td>RoBERTa</td>
      <td>0.98</td>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  <tr>
      <th>2.1</th>
      <td rowspan="3">20 Newsgroup</td>
      <td rowspan="3">Text</td>
      <td>BERT</td>
      <td>0.88</td>
      <td rowspan="3"><b>Task</b>: Topic Classification. <BR>
<b>Training Labels</b>: <i>Technology</i>, <i>Sale-Ads</i>, <i>Politics</i>, <i>Religion</i>, <i>Science</i> <BR>
<b>Drift</b>: Simulated with one new class label: <i>Recreation</i></td>
    </tr>
    <tr>
      <th>2.2</th>
      <td>DistillBERT</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>2.3</th>
      <td>RoBERTa</td>
      <td>0.88</td>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  <tr>
      <th>3.1</th>
      <td rowspan=2">STL</td>
      <td rowspan="2">Computer Vision</td>
      <td>VGG16</td>
      <td>0.89</td>
      <td rowspan="2"> <b>Task</b>: Image Classification. <BR>
<b>Training Labels</b>: <i>Forest</i>, <i>Glacier</i>, <i>Mountain</i>, <i>Building</i>, <i>Street</i> <BR>
<b>Drift</b>: Simulated with one new class label: <i>Sea</i></td>
    </tr>
    <tr>
      <th>3.2</th>
      <td>VisionTransformer</td>
      <td>0.90</td>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  <tr>
      <th>4.1</th>
      <td rowspan="2">STL</td>
      <td rowspan="2">Computer Vision</td>
      <td>VGG16</td>
      <td>0.82</td>
      <td rowspan="2"> <b>Task</b>: Image Classification. <BR>
<b>Training Labels</b>: <i>Airplane</i>, <i>Bird</i>, <i>Car</i>, <i>Cat</i>, <i>Deer</i>, <i>Dog</i>, <i>Horse</i>, <i>Monkey</i>, <i>Ship</i> <BR>
<b>Drift</b>: Simulated with one new class label: <i>Truck</i></td>
    </tr>
    <tr>
      <th>4.2</th>
      <td>VisionTransformer</td>
      <td>0.96</td>
    </tr>
    <tr class="separator">
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

## Offline Phase
The *offline* phase takes as input an historical dataset, called baseline, that represents what the model learned during training. Firstly, the baseline data is feed into the model to extract the embedding vectors and the predicted labels 
&#x2460;. Then, the majority of the baseline dataset is used to model the distributions of the baseline &#x2461;. Instead, a small portion of the baseline data is used to estimate the threshold values &#x2462;
## Online Phase

# DriftLens Demo

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
```sh
python driftlens_app.py
```
The DriftLens app will run on localhost: http://127.0.0.1:5000

# Recorded Demo
[![Watch the video](https://img.youtube.com/vi/1R2igFhMD8U&t=7s/hqdefault.jpg)](https://www.youtube.com/watch?v=1R2igFhMD8U&t=7s)



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

# People Involved

- **Salvatore Greco** - [Homepage](https://grecosalvatore.github.io/) - [GitHub](https://github.com/grecosalvatore) - [Twitter](https://twitter.com/_salvatoregreco)
- **Bartolomeo Vacchetti** - [Homepage]()
- **Daniele Apiletti** - [Homepage]()
- **Tania Cerquitelli** - [Homepage]()