# Deep Poisson Reconstruction

This is the official implementation of "GradNet: Unsupervised Deep Screened Poisson Reconstruction for Gradient-Domain Rendering" (SIGGRAPH Asia 2019).
For more details, please refer to [our paper](http://sites.cs.ucsb.edu/~lingqi/publications/paper_gradnet.pdf).

## Data

+ All data (dataset, testing scenes and model weights) can be downloaded from [here](https://pan.baidu.com/s/1O_7GLmNAHHD8gSxFT2ZrRQ) (access code: r71f).
+ The `gradnet_dataset.zip` is a multi-part archive. You can merge all parts by `zip -F gradnet_dataset.zip --out full_dataset.zip`, and then uncompress the `full_dataset.zip`.

## Usage

### Train

+ Make sure the dataset is located in `./dataset`.
+ Create a directory `./saved_models` to save model weights.
+ Run `python train.py`.

### Test

+ Make sure the testing scenes are located in `./test_data`.
+ Make sure `./saved_models` contains the model weight to test.
+ Run `python eval.py --epoch <epoch>`.

## Citation

If you find it useful in your research, please kindly cite our paper:

    @article{GradNet_SA2019,
    author = {Guo, Jie and Li, Mengtian and Li, Quewei and Qiang, Yuting and Hu, Bingyang and Guo, Yanwen and Yan, Ling-Qi},
    year = {2019},
    month = {11},
    pages = {1-13},
    title = {GradNet: unsupervised deep screened poisson reconstruction for gradient-domain rendering},
    volume = {38},
    journal = {ACM Transactions on Graphics},
    doi = {10.1145/3355089.3356538}
    }

## Contact

If you have any questions, please feel free to contact [guojie@nju.edu.cn](mailto:guojie@nju.edu.cn).
