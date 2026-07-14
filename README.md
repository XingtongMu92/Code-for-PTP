# Code and Data for Retrieval-Conditioned Probabilistic Trajectory Prediction for Maritime Early Warning Using AIS Sensor Data

This repository provides the reference implementation and datasets for the following paper:

> Xingtong Mu, Xueling Duan, and Ke Deng, “Retrieval-Conditioned Probabilistic Trajectory Prediction for Maritime Early Warning Using AIS Sensor Data,” submitted to *IEEE Sensors Journal*.

## 1. Repository Structure

```text
├── Code for the main program
│   ├── dataset_build.py
│   └── main.py
│
├── Code for comparative experiments
│   ├── baseline_CV_KF.py
│   ├── baseline_LSTM.py
│   ├── baseline_BiLSTM_Attn.py
│   ├── baseline_Prob_GRU.py
│   └── baseline_BiLSTM_MDN.py
│
├── Code for ablation experiment
│   ├── ablation_attention.py
│   └── ablation_retrieval.py
│
├── requirements.txt
└── README.md
```

## 2. Installation

Install the required Python dependencies using:

```bash
pip install -r requirements.txt
```

Please ensure that the required base libraries and the appropriate Python environment have been installed before running the code.

## 3. Raw AIS Data

The original AIS data are too large to be included in this repository. The data can be downloaded from the official MarineCadastre website:

https://marinecadastre.gov/ais/

Download the daily AIS files from **January 1 to January 20, 2025**, and place them in the working directory using the following filenames:

```text
ais-2025-01-01.txt
ais-2025-01-02.txt
...
ais-2025-01-20.txt
```

## 4. Dataset Construction

Place the downloaded raw AIS files in the same directory as `dataset_build.py`, and then run:

```bash
python dataset_build.py
```

This script constructs the datasets for the two study regions. The latitude and longitude ranges used for regional data extraction are described in the paper and in the comments within `dataset_build.py`.

## 5. Running the Experiments

Run the main experiment first:

```bash
python main.py
```

After completing the main experiment, run the comparative and ablation experiments as needed.

The comparative and ablation experiment scripts reuse components and configurations from `main.py`. Therefore, all experiment scripts should be placed under the same project directory, and the original repository structure should be preserved.

Example commands include:

```bash
python baseline_CV_KF.py
python baseline_LSTM.py
python baseline_BiLSTM_Attn.py
python baseline_Prob_GRU.py
python baseline_BiLSTM_MDN.py
```

For the ablation experiments, run:

```bash
python ablation_attention.py
python ablation_retrieval.py
```

## 6. Results

The compressed dataset files for the two study regions contain:

* reference data;
* test data; 
* prediction results.

The provided prediction results are consistent with those obtained by running the released code under the corresponding experimental settings.

## Acknowledgments and Contact

Thank you for your interest in our work. If you have any questions, suggestions, or encounter any issues while using the code or datasets, please feel free to contact us.
