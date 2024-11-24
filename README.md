# GRED

## Installation

1. Installation the package requirements

```
pip install -r requirements.txt
```

---

## Data Preparation

1. Download the tweet15, tweet16, and pheme datasets.

```
dataset/
├─ rumor/
│  ├─ twitter15.train.csv
│  ├─ twitter15.test.csv
│  ├─ ...
```

2. Download the fever datasets.

```
dataset/
├─ fever/
│  ├─ wiki-pages/
│  │  ├─ wiki-001.jsonl
│  │  ├─ wiki-002.jsonl
│  │  ├─ ...
```

3. Run the provided shell script (sh file) to preprocess the data.

```
sh evidence.sh
```

4. Run the provided shell script (sh file) to preprocess the data.

```
sh fever.sh
```

5. Run the provided shell script (sh file) to preprocess the data.

```
sh predict.sh
```

---

## Training

```
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
```

### You can also run the script

```
sh train.sh
```

---

## Contact

For any questions, welcome to create an issue or email to <a href="mailto:haoliangzhou6@gmail.com">2231984@s.hlju.edu.cn</a>.
