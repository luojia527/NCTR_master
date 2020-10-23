# NCTR

This is our implementation for the paper:


*Hybrid neural network combining textual information and rating information for item recommendation*



NCTR: This is the state-of-the-art method that uti-lizes deep learning technology to jointly model reviews for item recommendation.


## Environments

- python 3.5
- Tensorflow (version: 1.9.0)
- numpy
- pandas


## Dataset

In our experiments, we use the datasets from  Amazon 5-core(http://jmcauley.ucsd.edu/data/amazon) 
Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from Wikipedia 2014 + Gigaword 5 with 6B tokens tokens used for words.

## Example to run the codes		

Data preprocessing:

The implemention of data preprocessing is modified based on *[this](https://github.com/chenchongthu/NARRE)*


Train and evaluate the model:

```
python train.py
```



## Misc
The implemention of CNN is modified based on *[this](https://github.com/chenchongthu/NARRE)*




