# 训练命令

## 开始训练

```
cd /home/tml104/Mortal

tmux new-session -d -s mortal_transformer_bs1024_epoch1 \
  "exec env MORTAL_CFG=/home/tml104/Mortal/artifacts/mortal_transformer_offline_bs1024_v1/config.toml \
  /home/tml104/miniconda3/envs/mortal/bin/python -u \
  mortal/train_offline_grp_transformer_bs1024.py \
  >> artifacts/mortal_transformer_offline_bs1024_v1/console.log 2>&1"
```

## 状态查看

```
# 确认 tmux 会话存在
tmux ls

# 实时查看训练日志
tail -f artifacts/mortal_transformer_offline_bs1024_v1/train.log

# GPU
watch -n 2 nvidia-smi

# 进入训练终端
tmux attach -t mortal_transformer_bs1024_epoch1

```


## 停止训练

```
tmux send-keys -t mortal_transformer_bs1024_epoch1 C-c
```