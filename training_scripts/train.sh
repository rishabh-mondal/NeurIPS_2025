train_state="xxxx"
test_state="xxxx"
name="train_${train_state}_val_${test_state}"
type=sentinel
task=obb
suffix=dota
model_dir="./path/"
model=$model_dir/yolo11l-obb.pt
# model=best.pt
image_size=128
batch_size=16
save_period=100
epochs=100
device=1
base_path="./base_path"
data_path=$base_path/yaml_data_dir/train_test.yaml
val=True
# val_interval=10 #not supported in train mode 
save_conf=True
save_txt=False
experiment_name=$name\_$ratio\_$task\_$suffix\_$image_size\_$batch_size\_$epochs\_val_$val
log_file=$base_path/region_performance_logs/$experiment_name

echo "Name: $name"
echo "Task: $task"
echo "Ratio: $ratio"
echo "Suffix: $suffix"
echo "Model: $model"
echo "Image Size: $image_size"
echo "Batch Size: $batch_size"
echo "Epochs: $epochs"
echo "Device: $device"
echo "Base Path: $base_path"
echo "Data Path: $data_path"
echo "Val: $val"
echo "Save Conf: $save_conf"
echo "Save Txt: $save_txt"
echo "Save Period: $save_period"
echo "Log File: $log_file"
echo "Experiment Name: $experiment_name"

nohup yolo obb train model=$model\
    data=$data_path\
    imgsz=$image_size\
    epochs=$epochs\
    device=$device\
    val=$val\
    workers=8\
    batch=$batch_size\
    save_conf=$save_conf\
    save_txt=$save_txt\
    exist_ok=True\
    name=$base_path/runs_sentinel/obb/$experiment_name\
    save=True\
    > $log_file.log 2>&1 &

echo "Started training for $experiment_name"    
    

