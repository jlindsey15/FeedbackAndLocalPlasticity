import os

os.system("python3 retina_celltypes_color_regression.py --update_step 40 --meta_lr 0.0001 --update_lr 0.003 --tasks 10 --capacity 10 --width 300 --rln 0 --use_error --name MODEL_NAME --oja --overwrite --init_plasticity 0.0 --meta_plasticity_lr 1e-8 --iid --epoch 1000")

for i in range(0, 19):
    os.system("python3 retina_celltypes_color_regression.py --update_step 40 --meta_lr 0.0001 --update_lr 0.003 --tasks 10 --capacity 10 --width 300 --rln 0 --use_error --name MODEL_NAME --oja --overwrite --init_plasticity 0.0 --meta_plasticity_lr 1e-8 --iid --epoch 1000 --from_saved --model /path/to/model_file/MODEL_NAME_0/learner.model")


os.system("python3 evaluate_regression_oja.py --model /path/to/model_file/MODEL_NAME_0/learner.model --use_error --name EVAL_LOG_NAME --rln 0 --tasks 10 --capacity 10 --runs 50 --update_step 40 --meta_lr 0.0001 --update_lr 0.003 --width 300 --init_plasticity 0.0 --meta_plasticity_lr 1e-8 --iid")

