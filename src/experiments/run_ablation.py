"""
批量跑消融实验 A-C 组（基线）
C同学负责
"""
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from recbole.quick_start import run_recbole
import pandas as pd
import os

os.makedirs('results', exist_ok=True)

COMMON = {
    'data_path': 'data/processed',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
    'eval_args': {
        'split': {'RS': [0.8, 0.1, 0.1]},
        'order': 'TO',
        'mode': 'uni100'
    },
    'train_neg_sample_args': None,
    'topk': [10],
    'metrics': ['Recall', 'NDCG'],
    'valid_metric': 'NDCG@10',
}

# 消融A：ItemKNN
result_A = run_recbole(model='ItemKNN', dataset='taobao_sample', config_dict=COMMON)

# 消融B：GRU4Rec
result_B = run_recbole(model='GRU4Rec', dataset='taobao_sample', config_dict={
    **COMMON,
    'epochs': 30,
    'learning_rate': 0.001,
    'train_batch_size': 2048,
    'hidden_size': 128,
})

# 消融C：SASRec
result_C = run_recbole(model='SASRec', dataset='taobao_sample', config_dict={
    **COMMON,
    'epochs': 50,
    'learning_rate': 0.001,
    'train_batch_size': 2048,
    'hidden_size': 128,
    'n_heads': 2,
    'n_layers': 2,
    'attn_dropout_prob': 0.5,
    'hidden_dropout_prob': 0.5,
})

# 保存结果
rows = [
    {'组别': 'A', '模型': 'ItemKNN',
     'Recall@10': result_A['test_result']['recall@10'],
     'NDCG@10':   result_A['test_result']['ndcg@10']},
    {'组别': 'B', '模型': 'GRU4Rec',
     'Recall@10': result_B['test_result']['recall@10'],
     'NDCG@10':   result_B['test_result']['ndcg@10']},
    {'组别': 'C', '模型': 'SASRec',
     'Recall@10': result_C['test_result']['recall@10'],
     'NDCG@10':   result_C['test_result']['ndcg@10']},
]

df = pd.DataFrame(rows)
df.to_csv('results/baseline_results.csv', index=False)
print(df.to_string(index=False))