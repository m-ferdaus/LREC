# Leaky ReLU-based Evolving Classifier (LREC)
Activation functions (AFs) in deep neural networks (DNNs) and membership functions (MFs) in neuro-fuzzy systems (NFSs) play an important role in 
the performance of ML models. This work focuses on analysing the effects of various AFs/MFs in our developed online ML models while detecting defects in 
real-world nano-scaled semiconductor devices, where significant training samples are not available. From various semiconductor datasets having fewer samples, 
it has been observed that the proposed evolving neuro-fuzzy system (ENFS) with Leaky-ReLU MF performs better (improvement in the range of 1.8% to 23.5%
considering overall classification accuracy) than the other DNN or ENFS-based online ML models. In addition, the proposed model's performance has also been 
evaluated for handling large data streams problems and compared with some recently developed baselines under the prequential test-then-train protocol. 
The expected best classification rates are witnessed from the proposed model with an improvement in the range of 0.8% to 39.7%.

# How to Cite
Please cite the following work if you want to use LREC.

@misc{ferdaus2021significance ,
  title = {Significance of Activation Functions in Developing an Online Classifier for Semiconductor Defect Detection},
  author = {Ferdaus, Md Meftahul et.al.},
  year = {2021},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={Knowledge-Based Systems}
}

# LREC_Matlab

Significance of Activation Functions in Developing an Online Classifier for Semiconductor Defect Detection (accpeted for publication in Knowledge-Based Systems)

1. Clone LREC_Matlab git to your computer, or just download the files.
2. Open Matlab. The code was developed using Matlab 2016b, so if you use an older version, you might get some incompability errors. You can use Matlab 2016b or newer.
3. Execute the "secom_LREC.m" for SECOM dataset
4. Execute the follwoing files evaluate the code in prequential test-then-train settings for four different benchmark datasets:
   a) electricity_LREC_pre.m
   b) hyperplane_LREC_pre.m
   c) sea_LREC_pre.m
   d) weather_LREC_pre.m
   
   
# Datasets source: 
For electricity, hyperplane, sea, weather: https://github.com/ContinualAL/DEVFNN/tree/master/dataset
For SECOM: https://archive.ics.uci.edu/ml/datasets/SECOM  

Benchmark paper source:
https://github.com/ContinualAL/DEVFNN 
