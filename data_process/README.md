### 1.`data_clean.py` 
数据清洗：去重、去空、时间转换、ID映射、行为编码 `raw/UserBehavior.csv` `data_cleaned.csv` `id_mapping.pkl` 
- 从 chained.csv 中筛选需要的业务字段（user_id/item_id/behavior_type/timestamp 等）；
- 剔除无用冗余列，输出 chained_cleaned.csv，作为建图、模型输入的干净数据源。
### 2.`weight_calc.py` 
行为权重计算脚本：pv=1, cart=3, buy=5, fav=2 `data_cleaned.csv` `data_weighted.csv` 
- 区分 pv 浏览、cart 加购、buy 购买三种行为，分配不同交互权重；
- 输出带 weight_final 权重字段的完整行为链数据，给下游使用。
### 3.`need_chain.py` 
需求链识别：6小时截断，标记closure_label和need_state `data_weighted.csv` `chained.csv` 
- 读取用户原始行为表，按用户行为链（chain）聚合浏览 / 加购 / 购买行为；
- 生成 chained.csv，是整条流水线最原始的输入数据源。
### 4.`user_cluster.py` 
用户聚类：KMeans(K=4)，按行为特征分群 `chained.csv` `user_groups.pkl` 
### 5.`graph_build.py` 
构建稀疏图：pv/cart/buy三张用户-商品邻接矩阵 `chained.csv` `graph_pv.npz`, `graph_cart.npz`, `graph_buy.npz`  
- 读取清洗后的 csv，按 pv/cart/buy 拆分三类交互；
- 构建稀疏邻接矩阵，保存为 graph_pv/cart/buy.npz；
- 输出图卷积网络 GCN/GAT 所需的图结构文件。
### 6.`pipeline.py` 
流水线入口
生成交互文件：供后续推荐模型使用 `chained.csv` `taobao.inter` 
