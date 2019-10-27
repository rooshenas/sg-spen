This code runs SG-SPEN for the multi-label classification task. <br><br>

The code is in Python2.<br><br>

It has been written based on Tensorflow. <br><br>

Installing the requirements:<br>
pip install -r requirements.txt

Hyper-parameters:<br>
python ml-classification.py -h


usage: ml-classification.py [-h] [-lr [LEARNING_RATE]] [-ir [INF_RATE]] <br>
                            [-nr [NOISE_RATE]] [-it [INF_ITER]] <br>
                            [-mw [MARGIN_WEIGHT]] [-sm [SCORE_MARGIN]] <br>
                            [-l2 [L2_PENALTY]] [-dp [DROPOUT]] <br>
                            [-ip [INF_L2_PENALTY]] <br>
                           
<br>
optional arguments: <br>
  -h, --help            show this help message and exit <br>
  -lr [LEARNING_RATE]   Learning rate [0.001] <br>
  -ir [INF_RATE]        Inference rate (eta) [0.5] <br>
  -nr [NOISE_RATE]      Noise rate [2* eta] <br>
  -it [INF_ITER]        Inference iteration [10] <br>
  -mw [MARGIN_WEIGHT]   Margin (alpha) [100] <br>
  -sm [SCORE_MARGIN]    Reward Margin (delta) [0.01] <br>
  -l2 [L2_PENALTY]      L2 penalty [0.001] <br>
  -dp [DROPOUT]         Dropout [0.01] <br>
  -ip [INF_L2_PENALTY]  Inf L2 penalty [0.01] <br>

<br><br>

To execute the code with default params: <br>

cd multilabel <br>
python ml-classification.py

