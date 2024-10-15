#  Contextual Distribution Alignment via Correlation Contrasting for Domain Generalization


## Datasets

Our code supports the following dataset:

* [Office-31](https://github.com/jindongwang/transferlearning/tree/master/data#office-31)
* [Office-Home](https://github.com/jindongwang/transferlearning/tree/master/data#office-home)
* [Office-Caltech](https://github.com/jindongwang/transferlearning/tree/master/data#office-caltech10)
* [PACS](https://drive.google.com/uc?id=0B6x7gtvErXgfbF9CSk53UkRxVzg)
* [Digit-Five](https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/dg5.tar.gz)
* [VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)

If you want to use your own dataset, please organize your data in the following structure.

```
RootDir
└───Domain1Name
│   └───Class1Name
│       │   file1.jpg
│       │   file2.jpg
│       │   ...
│   ...
└───Domain2Name
|   ...    
```

And then, modifty `util/util.py` to contain the dataset.

## Usage

1. Modify the file in the scripts
2. The main script file is `train.py`, which can be runned by using `run.sh` from `scripts/run.sh`: `cd scripts; bash run.sh`.

## Customization

It is easy to design your own method following the steps:

1. Add your method (a Python file) to `alg/algs`, and add the reference to it in the `alg/alg.py`

2. Modify `utils/util.py` to make it adapt your own parameters

3. Midify `scripts/run.sh` and execuate it

Great thanks to [DeepDG](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG/).  Our code is based on this project and extends.


