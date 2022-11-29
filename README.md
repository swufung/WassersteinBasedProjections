# Wasserstein-based Projections
Pytorch implementation of Adversarial Projections.

## Associated Publication

_Wasserstein-Based Projections with Applications to Inverse Problems_ (**[arXiv Link](https://arxiv.org/abs/2008.02200)**)

Please cite as
    
    @article{heaton2022wasserstein,
      title={Wasserstein-Based Projections with Applications to Inverse Problems},
      author={Heaton, Howard and Fung, Samy Wu and Lin, Alex Tong and Osher, Stanley and Yin, Wotao},
      journal={SIAM Journal on Mathematics of Data Science},
      volume={4},
      number={2},
      pages={581--603},
      year={2022},
      publisher={SIAM}
    }

## Set-up

Install all the requirements:
```
pip install -r requirements.txt 
```

To run CT problems, first download the datasets from [here](https://drive.google.com/drive/folders/19ZDAutGypx4kkqMolLpSUUnN8C8JWcYd?usp=sharing).

The ellipse training and validation data should be stored in the following path: ./Datasets/

The Lodopab training and validation data should be stored in the following path: ./Datasets/ 


## Toy Problems

Two toy problems are included in this work. The first is an illustration of the "straggler" phenomenon about step size choices. We use

<img src="https://latex.codecogs.com/gif.latex?\lambda_k(u^k)=\mu_1\beta_k+\mu_2J_{\theta^k}(u^k)." /> 

If we can approximate the pointwise distance function <img src="https://latex.codecogs.com/gif.latex?d_{\mathcal{M}}" /> very well, then we will use a larger value of <img src="https://latex.codecogs.com/gif.latex?\mu_2" />  (e.g., choose <img src="https://latex.codecogs.com/gif.latex?\mu=(0,1)." />  in the ideal setting). If, however, we have limited ability to approximate <img src="https://latex.codecogs.com/gif.latex?d_{\mathcal{M}}" />, as is the case on more realistic problems like the CT examples, then we choose <img src="https://latex.codecogs.com/gif.latex?\mu=(0.5,0)" />. 

The second illustration uses a manifold that is a half circle. Here we clutter a region of interest uniformly and let our samples form our initial distribution <img src="https://latex.codecogs.com/gif.latex?\mathbb{P}^{1}" />. Then we conduct training for 19 steps to get <img src="https://latex.codecogs.com/gif.latex?\mathbb{P}^{20}" />. These results are then applied to solve a feasibility problem with 1 constraint, a line in the 2D plane. This makes for a nice illustration of the behavior of adversarial projections (try playing around with the diferent parameters, typically underrelaxation improves performance).

The [straggler illustration](https://colab.research.google.com/drive/1hhMmAr1MuBm9LOe29v8-cE88UdawUeRw?usp=sharing) and [toy manifold projection](https://colab.research.google.com/drive/1tO8T5E_Jycke9qV0s3uPsNqYybeDO0ue?usp=sharing) code can be run via these links online on Google Colab (or downloaded from this repo).

## Low Dose CT Problems

To train the ellipse CT problem run trainEllipseCT.py

To train the lodopab CT problem run trainLodopabCT.py

To deploy adversarial projections within projected gradient descent method, run deployProjection_Ellipse.py and deployProjection_Lodopab.py.

Note: these require the saved weights to be loaded.

## Acknowledgements

This material is partially funded by AFOSR MURI FA9550-18-1-0502, AFOSR Grant No. FA9550-18-1-0167, ONR Grants N00014-18-1-2527 snf N00014-17-1-21, and
NSF Grant No. DGE-1650604. 
Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF.




