from rfdetr import RFDETRBase, RFDETRLarge
import pandas as pd
import os

dataset_path = "./data/sentinel_data"
output_path = "./runs/sentinel_data"

# model = RFDETRBase()
model = RFDETRLarge()
history = []
def callback2(data):
    history.append(data)
model.callbacks["on_fit_epoch_end"].append(callback2)

model.train(
    dataset_dir=dataset_path,
    epochs=50,
    batch_size=20,
    grad_accum_steps=1,
    lr=2e-4,
    output_dir=output_path,
    tensorboard=True,
    # early_stopping=True
)

history_file = f"{output_path}/history.csv"
if os.path.exists(history_file):
    existing_history = pd.read_csv(history_file)
    updated_history = pd.concat([existing_history, pd.DataFrame(history)], ignore_index=True)
    updated_history.to_csv(history_file, index=False)
else:
    pd.DataFrame(history).to_csv(history_file, index=False)


## Training ##
# source .venv/bin/activate
# export CUDA_VISIBLE_DEVICES=0
# nohup python rfdetr_train.py > ./logs/rfdetr_large/sentinel_data.log 2>&1 &

## Tensorboard log ##
# tensorboard --logdir ./runs/sentinel_data 