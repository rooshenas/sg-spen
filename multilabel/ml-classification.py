
import sys
sys.path.append('../spen/')
import random
import sg_spen as sp, config
import argparse
import numpy as np
import os
import time
import pickle as pkl
import types
import tflearn.initializations as tfi

import tensorflow as tf
import tflearn

parser = argparse.ArgumentParser()
parser.add_argument('-lr', dest='learning_rate', nargs='?', help='Learning rate [0.001]')
parser.add_argument('-ir', dest='inf_rate', nargs='?', help='Inference rate (eta) [0.5]')
parser.add_argument('-nr', dest='noise_rate', nargs='?', help='Noise rate [2* eta]')
parser.add_argument('-it', dest='inf_iter', nargs='?', help='Inference iteration [10]')
parser.add_argument('-mw', dest='margin_weight', nargs='?', help='Margin (alpha) [100]')
parser.add_argument('-sm', dest='score_margin', nargs='?', help='Reward Margin (delta) [0.01]')
parser.add_argument('-l2', dest='l2_penalty', nargs='?', help='L2 penalty [0.001]')
parser.add_argument('-dp', dest='dropout', nargs='?', help='Dropout [0.01]')
parser.add_argument('-ip', dest='inf_l2_penalty', nargs='?', help='Inf L2 penalty [0.01]')

args = parser.parse_args()

parser.print_help(sys.stdout)

if args.l2_penalty:
    l2 = float(args.l2_penalty)
else:
    l2 = 0.001

if args.learning_rate:
    lr = float(args.learning_rate)
else:
    lr = 0.001

if args.inf_iter:
    it = float(args.inf_iter)
else:
    it = 10

if args.inf_rate:
    ir = float(args.inf_rate)
else:
    ir = 0.5

if args.score_margin:
    sm = float(args.score_margin)
else:
    sm = 0.01

if args.noise_rate:
    nr = float(args.noise_rate)
else:
    nr = 2*ir

if args.margin_weight:
    mw = float(args.margin_weight)
else:
    mw = 100.0

if args.inf_l2_penalty:
    ip = float(args.inf_l2_penalty)
else:
    ip = 0.01

if args.dropout:
    dp = float(args.dropout)
else:
    dp = 0.0



bs = 100

def f1_score_c_ar(cpred, ctrue):
  intersection = np.sum(np.minimum(cpred,ctrue),1)
  union = np.sum(np.maximum(cpred,ctrue),1)
  return np.divide(2.0*intersection, union + intersection)


def evaluate_score(xinput=None, yinput=None, yt=None):
    return np.array(f1_score_c_ar(yinput,yt) )

def check(xd, yd, yt=None):
    r = np.array(f1_score_c_ar(yd, yt))
    return np.average(r)


np.random.seed()


with open('data/bibtex-xdata.pkl', 'r') as f:
	xdata = pkl.load(f)

with open('data/bibtex-ydata.pkl', 'r') as f:
	ydata = pkl.load(f)

with open('data/bibtex-xtest.pkl', 'r') as f:
	xtest = pkl.load(f)

with open('data/bibtex-ytest.pkl', 'r') as f:
	ytest = pkl.load(f)

with open('data/bibtex-xval.pkl', 'r') as f:
	xval = pkl.load(f)

with open('data/bibtex-yval.pkl', 'r') as f:
	yval = pkl.load(f)

output_num = np.shape(ydata)[1]
input_num = np.shape(xdata)[1]



def get_energy_mlp(self, xinput=None, yinput=None, embedding=None, reuse=False):
        output_size = yinput.get_shape().as_list()[-1]
        with tf.variable_scope(self.config.spen_variable_scope):
            with tf.variable_scope(self.config.fx_variable_scope) as scope:
                net = xinput
                j = 0
                for (sz, a) in self.config.layer_info:
                    net = tflearn.fully_connected(net, sz,
                                                  weight_decay=self.config.weight_decay, activation=a,
                                                  weights_init=tfi.variance_scaling(),
                                                  bias_init=tfi.zeros(), regularizer='L2', reuse=reuse, scope=("fx.h" + str(j)))
                    net = tflearn.dropout(net, 1.0 - self.config.dropout)
                    j = j + 1
                logits = tflearn.fully_connected(net, output_size, activation='linear', regularizer='L2',
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(), bias_init=tfi.zeros(),
                                                  reuse=reuse, scope="fx.fc")



                mult = logits * yinput
                local_e = tf.reduce_sum(mult, axis=1)
            with tf.variable_scope(self.config.en_variable_scope) as scope:
                j = 0
                net = yinput
                for (sz, a) in self.config.en_layer_info:
                    net = tflearn.fully_connected(net, sz,
                                                  weight_decay=self.config.weight_decay,
                                                  weights_init=tfi.variance_scaling(),
                                                  activation=a,
                                                  bias=False,
                                                  reuse=reuse, regularizer='L2',
                                                  scope=("en.h" + str(j)))

                    j = j + 1
                global_e = tf.squeeze(tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                                   weights_init=tfi.variance_scaling(), bias=False,
                                                   reuse=reuse, regularizer='L2',
                                                   scope=("en.g")))



        return tf.squeeze(tf.add(local_e, global_e))




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
best_val_f1 = 0.0
test_f1 = 0.0

config = config.Config()
config.l2_penalty = l2
config.inf_iter = it
config.inf_rate = ir
config.learning_rate = lr
config.noise_rate = nr
config.margin_weight = mw
config.dropout = dp
config.dimension = 2
config.output_num = output_num
config.input_num = input_num
config.alpha = 1.0
config.inf_penalty = ip
config.en_layer_info = [(15, 'softplus')]
config.layer_info = [(1000,'relu')]
config.output_num = output_num
config.loglevel = 0
config.use_search = True
config.weight_decay = 0.0001
config.score_margin = sm
config.score_max = 1.0
s = sp.SPEN(config)
s.createOptimizer()
s.evaluate = evaluate_score
s.get_energy = types.MethodType(get_energy_mlp,s)
s.train_batch = s.train_unsupervised_sg_batch
s.construct(training_type=sp.TrainingType.Rank_Based)
print "Energy:"
s.print_vars()
s.init()


def search(self, xtest, yprev, yp, yt=None):
        final_best = np.zeros((xtest.shape[0], self.config.output_num))
        found_point = np.zeros(xtest.shape[0])
        checks = np.zeros(np.shape(xtest)[0])
        all_checks = np.zeros(np.shape(xtest)[0])
        propDic = {}
        total = 0
        for iter in range(np.shape(xtest)[0]):
            random_proposal = yprev[iter, :]
            if yt is not None:
                y = np.expand_dims(yt[iter], 0)
            else:
                y = None
            score_first = self.evaluate(np.expand_dims(xtest[iter], 0), np.expand_dims(random_proposal, 0), yt=y)
            best_score = np.copy(score_first[:])
            labelset = set(np.arange(self.config.dimension))
            found = False
            random_proposal_new = np.copy(random_proposal[:])
            
            n = 0
            propDic[iter] = []
            
            while n < 100 and not found:
                    
                    l = random.randint(0, self.config.output_num -1 )
                    label = random.randint(0, self.config.dimension -1 )
                    oldlabel = random_proposal_new[l]
                    random_proposal_new[l] = label
                    yprop = np.reshape(self.var_to_indicator(np.array([random_proposal_new])),
                                       (self.config.output_num * self.config.dimension))
                    ycurr = np.reshape(yp[iter, :], self.config.output_num * self.config.dimension)
                
                    
                  
                    all_checks[iter] += 1
  
                    score = self.evaluate(np.expand_dims(xtest[iter], 0),
                                          np.expand_dims(random_proposal_new, 0), yt=y)
                 
                    
                    if self.config.loglevel > 60:
                        print(iter, l, distance, score, score_first)

                    if score > best_score:
                        best_score = score

                        random_proposal_new[l] = label

                        if best_score > (score_first + self.config.score_margin) or (best_score + self.config.score_margin) > self.config.score_max: 
                            found = True
                            
                            total += 1
    
                    else:
                        random_proposal_new[l] = oldlabel

                    n += 1        

            found_point[iter] = 1 if found else 0
            if self.config.loglevel > 40:
                print("iter:", iter, "found:", found, "score first: ", score_first[0], "new score", best_score[0],
                      "all:", all_checks[iter])

            final_best[iter, :] = np.copy(random_proposal_new)
        return final_best, found_point
    
s.search = types.MethodType(search ,s)



total_num = np.shape(xdata)[0]
best_val = 0.0
test_val = 0.0
num_steps = 10000
best_vs = 0.0
best_ts = 0.0


start = time.time()
i = 1

while i < num_steps:
        b = 0
        perm = np.random.permutation(total_num)
        indices = perm[b * bs:(b + 1) * bs]
        xbatch = xdata[indices][:]
        ybatch = ydata[indices][:]

        noisex = np.random.normal(xbatch, np.random.uniform(0,2.5, np.shape(xbatch)[1])*np.std(xbatch,axis=0), size=np.shape(xbatch))
       

        s.set_train_iter(i)
        s.config.loglevel = 0
        
        o = s.train_batch(xbatch=noisex, ybatch=ybatch,verbose=1)

        s.config.loglevel = 0
       

        if i % 20 == 0:
            yts_out = s.map_predict(xtest)
            yval_out = s.map_predict(xval)
            ytr_out = s.map_predict(xdata)
            
            pe = check(xdata, ytr_out, yt=ydata)
            te = check(xtest, yts_out, yt=ytest)
            vs = check(xval, yval_out, yt=yval)
            if vs > best_vs:
                best_vs = vs
                best_ts = te
            stop = time.time()

	    print("Score: %d %0.3f %0.3f %d %0.3f %.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f" % (it, ir, ip, i, pe, vs, te, 0.0,np.mean(np.sum(yts_out,1)), best_vs, best_ts, stop-start))
            print("Val F1: %0.3f, Test F1: %0.3f, Best Test on Val: %0.3f" % (vs, te, best_ts))
        i+=1


