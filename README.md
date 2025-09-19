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
# 기존 알고리즘들 실행 DomainNet ver.
for ALGORITHM in 'IRM' 'GroupDRO' 'Mixup' 'MLDG' 'CORAL' 'MMD' 'DANN'; do
 for TEST_ENV in 0 1 2 3; do
   echo "Training $ALGORITHM on $DATASET with test environment $TEST_ENV"
   OUTPUT_DIR="/home/elicer/DomainBed/results/DomainNet/${ALGORITHM}/env${TEST_ENV}"
    mkdir -p $OUTPUT_DIR
    python -m domainbed.scripts.train \
      --dataset DomainNet \
      --algorithm=$ALGORITHM \
      --test_envs $TEST_ENV \
      --data_dir=$DATA_DIR \
      --output_dir=$OUTPUT_DIR \
      --steps=5000 \
      --checkpoint_freq=100 \
      --seed=1 \
      > output_${TEST_ENV}_DN_${ALGORITHM}.log 2> error_${TEST_ENV}_s1_${ALGORITHM}.log
  done
done
```
