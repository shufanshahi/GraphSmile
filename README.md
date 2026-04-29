# GraphSmile
The official implementation of the paper "[Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for multimodal emotion recognition](https://doi.org/10.1109/TPAMI.2025.3581236)", which has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).  
Authors: Jiang Li, Xiaoping Wang, Zhigang Zeng  
Affiliation: Huazhong University of Science and Technology (HUST)  

## Citation
```bibtex
@article{li2025tracing,
    title={Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for multimodal emotion recognition},
    author={Jiang Li and Xiaoping Wang and Zhigang Zeng},
    year={2025},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    volume={47},
    number={10},
    pages = {8786-8803},
    doi={10.1109/TPAMI.2025.3581236}
}
```

## Requirement
Checking and installing environmental requirements
```python
pip install -r requirements.txt
```
## Datasets
链接: https://pan.baidu.com/s/1u1efdbBV3HP8FLj3Gy1bvQ
提取码: ipnv
Google Drive: https://drive.google.com/drive/folders/1l_ex1wnAAMpEtO71rjjM1MKC7W_olEVi?usp=drive_link

Adding the dataset path to the corresponding location in the run.py file, e.g. IEMOCAP_path = "".

## Run
### IEMOCAP-6
```bash
python -u run.py --gpu 2 --port 1530 --classify emotion \
--dataset IEMOCAP --epochs 120 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 \
--win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7
```

### IEMOCAP-4
```bash
python -u run.py --gpu 2 --port 1531 --classify emotion \
--dataset IEMOCAP4 --epochs 120 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 3e-04 --batch_size 16 --hidden_dim 256 \
--win 5 5 --heter_n_layers 4 4 4 --drop 0.2 --shift_win 10 --lambd 1.0 0.6 0.6
```

### MELD
```bash
python -u run.py --gpu 2 --port 1532 --classify emotion \
--dataset MELD --epochs 50 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 7e-05 --batch_size 16 --hidden_dim 384 \
--win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2
```

### CMUMOSEI
```bash
python -u run.py --gpu 3 --port 1534 --classify emotion \
--dataset CMUMOSEI7 --epochs 60 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 8e-05 --batch_size 32 --hidden_dim 256 \
--win 5 5 --heter_n_layers 2 2 2 --drop 0.4 --shift_win 2 --lambd 1.0 0.8 1.0
```

## EACL (Emotion-Anchored Contrastive Learning) Integration

GraphSmile+EACL adds emotion-anchored contrastive learning to improve discrimination between similar emotions (excited/happy, frustrated/angry). Uses a two-stage training: Stage 1 trains all params with contrastive + multi-task losses; Stage 2 adapts anchor vectors only for cosine-space classification.

### IEMOCAP-6 + EACL
```bash
python -u run_eacl.py --gpu 0 --port 1540 --classify emotion \
--dataset IEMOCAP --stage1_epochs 114 --stage2_epochs 6 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 1e-04 --stage2_lr 1e-3 --batch_size 16 --hidden_dim 512 \
--win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7 \
--lambda1 0.9 --lambda2 0.01 --temperature 0.1
```

### IEMOCAP-4 + EACL
```bash
python -u run_eacl.py --gpu 0 --port 1541 --classify emotion \
--dataset IEMOCAP4 --stage1_epochs 114 --stage2_epochs 6 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 3e-04 --stage2_lr 1e-3 --batch_size 16 --hidden_dim 256 \
--win 5 5 --heter_n_layers 4 4 4 --drop 0.2 --shift_win 10 --lambd 1.0 0.6 0.6 \
--lambda1 0.9 --lambda2 0.01 --temperature 0.1
```

### MELD + EACL
```bash
python -u run_eacl.py --gpu 0 --port 1542 --classify emotion \
--dataset MELD --stage1_epochs 44 --stage2_epochs 6 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 7e-05 --stage2_lr 1e-3 --batch_size 16 --hidden_dim 384 \
--win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2 \
--lambda1 0.1 --lambda2 0.1 --temperature 0.1
```

**Key EACL arguments:**
- `--lambda1`: weight of contrastive term vs CE loss (0.9 for IEMOCAP, 0.1 for MELD)
- `--lambda2`: weight of anchor angle loss within contrastive term (0.01–0.1)
- `--temperature`: τ for cosine similarity scaling (0.1–0.15)
- `--stage1_epochs`: epochs for representation learning with contrastive losses
- `--stage2_epochs`: epochs for anchor adaptation (anchors-only training)
- `--stage2_lr`: learning rate for anchor adaptation (typically higher than stage 1)