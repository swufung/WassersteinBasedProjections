# Adversarial Projections
Pytorch implementation of Adversarial Projections.

## Associated Publication

Projecting to Manifolds via Unsupervised Learning (**[arXiv Link](https://arxiv.org/abs/2008.02200)**)

Please cite as
    
    @article{heaton2020projecting,
        title={Projecting to Manifolds via Unsupervised Learning},
        author={Heaton, Howard and Fung, Samy Wu and Lin, Alex Tong and Osher, Stanley and Yin, Wotao},
        journal={arXiv preprint arXiv:2008.02200},
        year={2020}}

## Set-up

Install all the requirements:
```
pip install -r requirements.txt 
```

To run CT problems, first download the datasets from [here](https://drive.google.com/drive/folders/19ZDAutGypx4kkqMolLpSUUnN8C8JWcYd?usp=sharing).

The ellipse training and validation data should be stored in the following path: ./CTEllipse/experimentsEllipse/

The Lodopab training and validation data should be stored in the following path: ./CTLodopab/experimentsLodopab/ 


## Toy Problems

The [straggler illustration](https://colab.research.google.com/drive/1hhMmAr1MuBm9LOe29v8-cE88UdawUeRw?usp=sharing) and [toy manifold projection](https://colab.research.google.com/drive/1tO8T5E_Jycke9qV0s3uPsNqYybeDO0ue?usp=sharing) code can be run via these links online on Google Colab.

## Low Dose CT Problems

To train the ellipse CT problem run trainEllipseCT.py

To train the lodopab CT problem run trainLodopabCT.py

To deploy adversarial projections within projected gradient descent method, run deployProjection_Ellipse.py and deployProjection_Lodopab.py.

Note: these require the saved weights to be loaded.

## Acknowledgements

This material is partially funded by AFOSR MURI FA9550-18-1-0502, AFOSR Grant No. FA9550-18-1-0167, ONR Grants N00014-18-1-2527 snf N00014-17-1-21, and
NSF Grant No. DGE-1650604. 
Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF.




