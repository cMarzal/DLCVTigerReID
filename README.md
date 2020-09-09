# Tiger ReID in the Wild
## Final Project of the course Deep Learning and Computer Vision 2019 at National Taiwan University 

In collaboration with Céline Nauer, Javi Sanguino and Julia Maricalva

![illustration](https://cvwc2019.github.io/imgs/det_1.jpg)

## Introduction:

Tiger re-identification aims to find all image in the database containing the same tiger as the query one.

The original challenge website is provided [here](https://cvwc2019.github.io/challenge.html).

## Dataset 
One can download the dataset [here](https://drive.google.com/file/d/1QmvUBz07IphyIi-80iz5B5ZWMEC0IrSq/view?usp=sharing).

The dataset arrangment is listed as follow: 
* **imgs/** : All tiger images (* .jpg).
* **train.csv** : Image names and corresponding ids for training.
* **query.csv** : Image names and corresponding ids of query images.
* **gallery.csv** : Image names and corresponding ids of gallery images.

## Instructions to run the code: 

### Packages

You can run the following command to install all the packages (in python 3.6) listed in the requirements.txt:

    pip3 install -r requirements.txt


The code can be executed in the following manner

```
CUDA_VISIBLE_DEVICES=GPU_NUMBER bash final.sh $1 $2 $3 $4
```
* `$1` is the image folder (e.g. `imgs/`)of testing images.
* `$2` is the path to query.csv (e.g. `query.csv`). Please note that when TA's run your code, the ground truth id will not be available in the query.csv
* `$3` is the path to support.csv (e.g. `gallery.csv`). Please note that when TA's run your code, the ground truth id will not be available in the query.csv
* `$4` is the path to the csv file to save your predicted nearest neighbor for each quesry image.(e.g. results.csv)

The predicted csv file (e.g. results.csv) should be in the following format:

| img_name |
|:-----:|
XXXXX.jpg |
YYYYY.jpg |
ZZZZZ.jpg |
WWWWW.jpg |

where the ith row is the top1 retrieval corresponds to the ith row in query.csv.

In order to evaluate the generated result using Rank1, the provided script (by NTU TA's) can be run as follows: 

```
python3 evaluate.py --query $1 --gallery $2 --pred $3
```
* `$1` : the path to query.csv (e.g. `query.csv/`).
* `$2` : the path to gallery.csv (e.g. `gallery.csv`).
* `$3` : the path to the predicted csv file. (e.g. `results.csv`)
