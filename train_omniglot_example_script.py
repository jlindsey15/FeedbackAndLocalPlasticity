import os

os.system("python3 run_training_classification.py --rln 6   --tasks 5 --update_step 25 --steps 40000 --oja --name MODEL_NAME --meta_plasticity_lr 1e-5 --init_plasticity 0.0  --overwrite --iid --train_on_new --seed 24 --feedback_strength 1.0  --meta_feedback_strength_lr 0.0 --inner_plasticity_multiplier 100")


os.system("python3 evaluate_classification_oja.py   --rln 6  --modelX /pqth/to/model/file/MODEL_NAME --name EVAL_LOGS_NAME  --test --runs 500 --shots 5 --iid --fiveclass")

