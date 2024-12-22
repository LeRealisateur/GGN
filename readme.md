# Benchmarking Deep Learning Models Performance for Raw EEG Classification: Separating Noise from Brain Activity

Enhancing the understanding of how deep learning models separate genuine brain activity from noise in EEG signals.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [License](#license)

## Introduction
Electroencephalography (EEG) is widely used in neuroscience and clinical diagnostics but is prone to noise from external 
and internal sources. Deep learning models offer opportunities to learn directly from raw EEG data, bypassing traditional
preprocessing steps. However, these models face challenges related to interpretability and robustness. 
This project addresses these issues by evaluating deep learning models under noisy and clean conditions, with a focus 
on understanding their decision-making processes.

## Features
- Highlight key features of the project.
- Include any significant aspects worth noting.

## Installation
The first cell of the notebook will install all the packages needed to execute 
the code in this project.

## Dataset
The data can be downloaded at the drive linked provided in the instructions.
The raw folder as to be placed in a data folder as such: data/raw. To preprocess and generate the needed data,
you will need to execute the Preprocessing cell in the project_pipeline.ipynb.
This script will first execute the preprocessing for extracting relevant data linked
to the needed task and split the recording into multiple epochs keeping only electrodes
used for eeg data. Then, the data will be plit into training, validation and testing data.
The results will be in split/train,val,test folders. The command will then call the noise generation
script so the epochs generated by the previous script will have gaussian noise applied to them.
The same patter will be used to split them. The epochs will be situated in the split_noisy/train,val,test folders.

## Configuration
Configurations can be created to fit your specific model's needs. As shown in the 
notebook, each configuration must be created and loaded before executing the model's training. 
The following lines:

```python
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
```

load the specific configurations for the model.

If you want to execute a specific mode, the line:

```yaml
running_model: ""
```

needs to be adapted based on your requirements. Possible choices include:
- `'GGN'`
- `'SVM'`
- `'EEGNet'`
- `'EEGDeformer'`

Ensure the correct mode is set to match the model you intend to train or evaluate.

The line:

```yaml
split_data_save_path: "data/split_noisy"
```

also needs to be adjusted based on the type of data you're using. 
The possible choices are either `"data/split"` for clean data or `"data/split_noisy"` for noisy data.

Finally, 

```yaml
results_save_path: "results/"
```
can be changed depending on where you want to save the resulting explainability results for the GGN model.

### Usage

The cells bellow each configuration loading cell can then be executed to train and 
evaluate the different models.

## Licence

MIT License

Copyright (c) 2024 LeRealisateur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

