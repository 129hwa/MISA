# MISA: Mutual Information-driven Separator with Spectral Alignment

Pytorch implementation


This project is built upon facebook@DomainBed and khanrc@MIRO.

## Preparation

### Dependencies
```python
pip install -r requirements.txt
```

### Datasets
```python
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### How to Run
```python
for TEST_ENV in 0 1 2 3; do
    OUTPUT_DIR="/my/results/path"
    mkdir -p $OUTPUT_DIR
    python -m domainbed.scripts.train \
        --dataset PACS \
        --algorithm=MISA \
        --test_envs $TEST_ENV \
        --data_dir=$DATA_DIR \
        --output_dir=$OUTPUT_DIR \
        --seed=0 \
    > output.log 2> error.log
done
```
