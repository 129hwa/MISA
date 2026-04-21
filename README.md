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
python -m domainbed.scripts.sweep launch \
      --data_dir "$DATA_DIR" \
      --output_dir "$SWEEP_DIR" \
      --command_launcher local \
      --algorithms MISA \
      --datasets VLCS \
      --n_hparams 1 \
      --n_trials 1 \
      --steps 4000 \
      --seed 0 \
      --skip_confirmation \
      2>&1 | tee "${SWEEP_OUTPUT_DIR}/sweep_log.log"
```
