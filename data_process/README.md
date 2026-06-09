### 1.`data_clean.py` 
数据清洗：去重、去空、时间转换、ID映射、行为编码 `raw/UserBehavior.csv` `data_cleaned.csv` `id_mapping.pkl` 
### 2.`weight_calc.py` 
权重计算：pv=1, cart=3, buy=5, fav=2 `data_cleaned.csv` `data_weighted.csv` 
### 3.`need_chain.py` 
需求链识别：6小时截断，标记closure_label和need_state `data_weighted.csv` `chained.csv` 
### 4.`user_cluster.py` 
用户聚类：KMeans(K=4)，按行为特征分群 `chained.csv` `user_groups.pkl` 
### 5.`graph_build.py` 
构建稀疏图：pv/cart/buy三张用户-商品邻接矩阵 `chained.csv` `graph_pv.npz`, `graph_cart.npz`, `graph_buy.npz`  
### 6.`pipeline.py` 
生成交互文件：供后续推荐模型使用 `chained.csv` `taobao.inter` 

