
## 📝 Environment

You can run the following command to download the codes faster:

```bash
git clone https://github.com/HKUDS/SA-GNN.git
```

Then run the following commands to create a conda environment:

```bash
conda create -y -n sagnn python=3.6.12
conda activate sagnn
pip install matplotlib==3.5.1
pip install numpy==1.21.5
pip install scipy==1.7.3
pip install tensorflow_gpu==1.14.0
```

## 📚 Recommendation Dataset

I utilized a public datasets to evaluate: Amazon. Following the common settings of implicit feedback, if user  has rated item , then the element  is set as 1, otherwise 0. I filtered out users and items with too few interactions.


The datasets are in the `./Dataset` folder:

```
- ./Dataset/amazon(yelp/movielens/gowalla)
|--- sequence    # user behavior sequences (List)
|--- test_dict    # test item for each users (Dict)
|--- trn_mat_time    # user-item graphs in different periods (sparse matrix)
|--- tst_int    # users to be test (List)
```

### Original Data

The original data of our dataset can be found from following links

- Amazon-book: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html


## 🚀 Examples to run the codes

You need to create the `./History/` and the `./Models/` directories. The command to train SA-GNN on the Gowalla/MovieLens/Amazon/Yelp dataset is as follows.

- Amazon

```
./amazon.sh > ./amazon.log 

```

**Thanks for your interest in our work!**
