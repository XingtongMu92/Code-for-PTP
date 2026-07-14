# Code and Data for Retrieval-Conditioned Probabilistic Trajectory Prediction for Maritime Early Warning Using AIS Sensor Data," submitted to *IEEE Sensors Journal*.
This repository contains the reference implementation for the paper:

> Xingtong Mu, Xueling Duan, and Ke Deng, "Retrieval-Conditioned Probabilistic Trajectory Prediction for Maritime Early Warning Using AIS Sensor Data," submitted to *IEEE Sensors Journal*.

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

> 2. Installation\
>    The requirements.txt file provides all dependencies except for the base library. Please use the pip command for installation
> 3. Raw AIS data\
>    The original data is too large to be displayed here. Please visit the official website of MarineCadastre:https://marinecadastre.gov/ais/.
>    Download the daily AIS files for **January 1–20, 2025**, and place them in the working directory as:

```
ais-2025-01-01.txt
ais-2025-01-02.txt
...
ais-2025-01-20.txt
```
> 4. Build the two regional datasets\
>    Please place the raw data in the same folder as the dataset-build. py program and run the dataset building program. Please refer to the main text and program comments for specific latitude and longitude.
> 5. Experiment\
>    Please prioritize running main. py before conducting comparative and ablation experiments. Please note that the code for the comparison and ablation experiments is based on main.exe. Please place them in one folder.
> 6. Results\
>    The zip files of our two datasets provide their respective references, test data, and test results. This is consistent with the results obtained by running our code.
