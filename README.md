# Learning to Learn with Feedback and Local Plasticity

This code implements experiments described in the paper "Learning to Learn with Feedback and Local Plasticity", to be presented at NeurIPS 2020. 

The scripts train_regression_example_script.py and train_omniglot_example_script.py launch meta-training for the regression and Omniglot experiments, respectively, using default parameters.

To customize network and meta-learning parameters, you can modify these scripts by varying the keyword arguments below:

update_step: in the regression task, this sets the number of inner loop steps per sinusoid.  In the classification task, it sets the total number of inner loop steps

tasks: number of sinusoids (for regression), or Omniglot classes (for classification)

meta_lr: meta learning rate of feedforward initialization

meta_feedback_lr: meta learning rate of feedback weights

width: width of MLP layers.

rln: number of frozen (during inner loop) layers, starting from front

rln_end: number of frozen (during inner loop) layers, starting from end [I'm not sure the code is still compatible with using this keyword]

use_error: during feedback, propagate y - yhat rather than just y

name: what you want to call the model

oja: use Oja's rule (replace with "hebb" if you want hebb's rule, or neither if you want gradient-based)

overwrite: overwrite existing saved files corresponding to this model name.  Otherwise, will create a new file by appending a suffix ("modelname_1", "modelname_2", etc)

init_plasticity: initial plasticity coefficients of all synapses

meta_plasticity_lr: meta learning rate of plasticity coefficients

iid: if set, inner loop task is i.i.d.  Otherwise, the continual learning version of the task is used.

epoch: number of meta-training steps

from_saved: if set, load model from saved model

model: if "from_saved" is set, this gives the model file location
