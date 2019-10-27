This code runs SG-SPEN for the multi-label classification task.

The code is in Python2.

It has been written based on Tensorflow.

Installing the requirements:
pip install -r requirements.txt

Hyper-parameters:
python ml-classification.py -h

usage: ml-classification.py [-h] [-lr [LEARNING_RATE]] [-ir [INF_RATE]]
                            [-nr [NOISE_RATE]] [-it [INF_ITER]]
                            [-mw [MARGIN_WEIGHT]] [-sm [SCORE_MARGIN]]
                            [-l2 [L2_PENALTY]] [-dp [DROPOUT]]
                            [-ip [INF_L2_PENALTY]]

optional arguments:
  -h, --help            show this help message and exit
  -lr [LEARNING_RATE]   Learning rate [0.001]
  -ir [INF_RATE]        Inference rate (eta) [0.5]
  -nr [NOISE_RATE]      Noise rate [2* eta]
  -it [INF_ITER]        Inference iteration [10]
  -mw [MARGIN_WEIGHT]   Margin (alpha) [100]
  -sm [SCORE_MARGIN]    Reward Margin (delta) [0.01]
  -l2 [L2_PENALTY]      L2 penalty [0.001]
  -dp [DROPOUT]         Dropout [0.01]
  -ip [INF_L2_PENALTY]  Inf L2 penalty [0.01]


To execute the code with default params: 

python ml-classification.py

