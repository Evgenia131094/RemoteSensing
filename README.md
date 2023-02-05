**Project structure:**
```
components
|
|--config:
|  |--config_classes.py
|
|--Dataset:
|  |--dataset.py
|
|--docs
|  |--QuickStart.md
|
|--Preprocessing:
|  |--preprocessing.py
|
|--Training
|  |--Classification
|  |  |--classification.py
|  |
|  |--Clusterization
|  |  |--clusterization.py
|  |
|  |--constants.py
|  |--train.py
|
|--Utils
|  |--utils.py
|
|--info.py
|--start.py <-entry point
```

App modes:
1) Preprocessing:
   1) Generation annotations.csv(each row describe one image pixel) based on input dataset and config data.
       **Example of columns names:** image,class,Aerosol,Blue,Green,Red,IR1,IR2,IR3,IR4,IR5,IR6,IR7,IR8
   2) Basic dataset info:
      1) Correlation matrix 
      2) Histograms for each feature
      3) Pair plots

2) Clusterization
   1) Fitting set clusterization algorithms
   2) Visualisation results with Umap and plots by two dimensions