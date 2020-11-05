# diseaseBERT
Code and dataset of EMNLP 2020 paper "Infusing Disease Knowledge into BERT for Health Question Answering, Medical Inference and Disease Name Recognition"

Paper link: https://arxiv.org/pdf/2010.03746.pdf

Author homepage: http://people.tamu.edu/~yunhe/

In disease_knowledge_infusion_training.py, we show how to infuse diseae knowledge into BERT.
You could change BERT to other BERT-like models mentioned in our paper: 

Albert (https://huggingface.co/albert-xxlarge-v2)

ClinicalBERT (https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT)

SicBERT (https://huggingface.co/allenai/scibert_scivocab_uncased)

BioBERT (https://github.com/dmis-lab/biobert)

BlueBERT (https://github.com/ncbi-nlp/bluebert)

We use pytorch-based Huggingface BERT but BioBERT and BlueBERT are based on tensforflow. For them, we use the method from this blog (https://medium.com/@manasmohanty/ncbi-bluebert-ncbi-bert-using-tensorflow-weights-with-huggingface-transformers-15a7ec27fc3d) to transfrom tensorflow-based models into pytorch-based version.

Package:
Python 3.6.8, Pytorch 1.4.0, Huggingface transformers 2.5.1
