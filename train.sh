python rumor.py \
--config 'config/model.yaml' \
--mode 'model' \
--bert 'roberta-base' \
--bart 'facebook/bart-large' \
--output_dir 'result' \
--log_dir 'log' \
--checkpoint 'rumor.pth' \
--dataset 'pheme' \
--seed 42 \
--epochs 40 \
--folds 5