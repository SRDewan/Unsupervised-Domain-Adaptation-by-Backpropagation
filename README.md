# Unsupervised-Domain-Adaptation-by-Backpropagation

## Setup Instructions
```
conda env create -f environment.yml
```

## Run Instructions
All the code is in jupyter notebooks where we have created a separate notebook for each datset pair. Thus, the code is self-explanatry to run, just run the notebooks' cells.

## Codebase
1. datasets/syn_signs_loader.py - Contains the torch dataset loader for the SYN SIGNS dataset.
2. models/utils.py - Contains the definition of the GRL layer.
3. models/gtsrb_cnn.py - Contains the definiton of the base CNN model used for the SYN SIGNS --> GTSRB pair.
4. gtsrb.ipynb - Contains training and evaluation code for the SYN SIGNS --> GTSRB dataset pair including source-only and target-only baselines. It also contains all the runs and results for the various experiments conducted of fixed alpha, fixed lr, exponential lr, mean loss and semi-supervised setting.
5. gtsrb_vis.ipynb - Contains feature visualization code and results for the SYN SIGNS --> GTSRB models.
6. checkpoints/ - Contains the checkpoints/saved models for some of the runs/experiments, not all (due to size issues).
7. `syn_mnist.ipynb` - Contains code for the experiments where Syn Numbers is the source and MNIST is the target. 
8. `syn_svhn.ipynb` - Contains code for the experiments where Syn Numbers is the source and SVHN is the target. 
9. `office_aw.ipynb` - Contains code for the Office-31 dataset experiments where Amazon is the source and Webcam is the target. 
10. `office_dw.ipynb` - Contains code for the Office-31 dataset experiments where DSLR is the source and Webcam is the target. 
11. `office_wd.ipynb` - Contains code for the Office-31 dataset experiments where Webcam is the source and DSLR is the target. 
12.`mnist_mnistm(updated).ipynb`- Contains code for the Mnist dataset as source and Mnist_M dataset as target.


## Datasets
The MNIST and GTSRB datasets were directly taken from torchvision. The rest of the datasets have been uploaded to a drive (link given below).

## Links
[Drive Link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/abhiroop_talasila_research_iiit_ac_in/EntcFxh6NSZOq84pWX1mp9gBxH4wJFUMlfWxR5P9l3tbeg?e=4WCRha) - Contains some of the datasets and all the checkpoint files.
