# scLDS2
Learning Discriminative and Structural Samples for Rare Cell Types with Deep Generative Model
## Dependencies 
The code has been tested with the following versions of packages.
- Python 2.7.12
- Tensorflow 1.4.0
- Numpy 1.14.2

###Guiding principles:
**We only provide one single-cell RNA-seq dataset, other datasets can be obtained from corresponding website.
##Datasets
.scLDS2/data/<dataset_name>
GSM2230759_human3.csv - Braon Human3 dataset expression data. 
Label- The cell type of Human3 dataset.

##Training
You can either train your own models on the datasets or use pre-trained models (--train = False) in pre_trained_models/<dataset_name>. This will save the model along with timestamp in checkpoint-dir/<dataset_name>.

**scLDS2_model: /clus_wgan.py and Baron.py are the implementation of scLDS2 networks.

#Files:
Baron.py - A script with a real scRNA-seq data to shows how to run the code.
Metric.py – This is the implementation of NMI,ARI,ACC.
rare data acc.py – the identification of rare cell types
informative gene.py-informative genes analysis.

Follow the steps below to run scLDS2（also contained in the " Baron.py" file）. Here use a real scRNA-seq data (Baron_Human3) set as an example.

Input: The input of scLDS2 is the data normalized by TPM to scRNA-seq count data.
Output: label_predicted.csv and cell features.csv and Three evalustion indexes: NMI,ARI,ACC


