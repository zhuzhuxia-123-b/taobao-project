import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import os

INPUT_FILE = r"D:\\社会计算\\taobao-project\\data\\processed\\data_weighted.csv"
OUTPUT_FILE = r"D:\\社会计算\\taobao-project\\data\\processed\\chained.csv"
TEMP_FILE = r"D:\\社会计算\\taobao-project\\data\\processed\\temp_chained.csv"
CHUNK_SIZE = 1000000

if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)
    print("🗑️ 已删除旧的临时文件")

print("⏳ 开始处理数据...")
total_rows = 0

for i, chunk in enumerate(tqdm(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE), desc="处理数据块")):
    chunk["behavior_code"] = pd.to_numeric(chunk["behavior_code"], errors="coerce").fillna(0).astype(int)
    chunk = chunk.sort_values(["user_id", "timestamp"])
    
    result_rows = []
    for uid, group in chunk.groupby("user_id"):
        group = group.sort_values("timestamp")
        records = group.to_dict("records")
        
        chains = []
        current_chain = []
        current_time = None
        
        for record in records:
            t = record["timestamp"]
            if current_time is None:
                current_chain = [record]
                current_time = t
            else:
                if t - current_time > 6 * 3600:
                    chains.append(current_chain)
                    current_chain = [record]
                else:
                    current_chain.append(record)
                current_time = t
        
        if current_chain:
            chains.append(current_chain)
        
        for chain_id, chain in enumerate(chains):
            has_buy = any(r["behavior_code"] == 3 for r in chain)
            has_cart = any(r["behavior_code"] == 2 for r in chain)
            action_count = len(chain)
            
            times = [r["timestamp"] for r in chain]
            max_gap = max(times[i+1] - times[i] for i in range(len(times)-1)) if len(times) > 1 else 0
            
            if max_gap > 24 * 3600:
                need_state = "dormant"
                closure_label = 0
            elif has_buy:
                need_state = "closed"
                closure_label = 1
            elif has_cart:
                need_state = "intent"
                closure_label = 0
            elif action_count >= 2:
                need_state = "exploring"
                closure_label = 0
            else:
                need_state = "latent"
                closure_label = 0
            
            for record in chain:
                record["chain_id"] = chain_id
                record["closure_label"] = closure_label
                record["need_state"] = need_state
                result_rows.append(record)
    
    block_result = pd.DataFrame(result_rows)
    total_rows += len(block_result)
    
    block_result.to_csv(TEMP_FILE, mode="a", header=(i == 0), index=False)
    
    del block_result, chunk, result_rows
    gc.collect()

print(f"\n✅ 分块处理完成！共 {total_rows:,} 行")
print("⏳ 转换为目标格式...")

df = pd.read_csv(TEMP_FILE)
os.remove(TEMP_FILE)

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ need_chain 完成！输出文件: {OUTPUT_FILE}")
