## WSCBS2021 Preprocessing

This implementation is based on [bogorodvo's solution](https://www.kaggle.com/bogorodvo/lightgbm-baseline-model-using-sparse-matrix).

This is an example Brane package for a preprocessing solution. Import it as follows:

```shell
$ brane import yaaani85/wscbs2021-preprocessing
```

The following environment variables can be set: 

```shell
$ export USE_LOCAL=True 
$ export USE_SAMPLED_DATA=True
```

For an overview of the parameters of the brane package, you can `test` the package
```shell
$ brane --debug test preprocessing
```


You also need to push the package to be able to import it in your remote session or jupyterlab notebook:
```shell
brane push preprocessing 1.0.0
```
