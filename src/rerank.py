import pandas as pd

CHAINED_CSV_PATH = "D:/社会计算/data/chained.csv"

class DemandRerank:
    def __init__(self):
        self.strategy_map = {
            "latent": {"popular": 0.7, "similar": 0.3},
            "exploring": {"similar": 0.6, "fresh": 0.4},
            "intent": {"click_seq": 0.8, "cart": 0.2},
            "dormant": {"recall": 0.5, "hot": 0.5}
        }
        self.valid_state = list(self.strategy_map.keys())

    def judge_user_state(self, user_df: pd.DataFrame) -> str:
        total_pv = len(user_df[user_df["behavior_type"] == 0])
        total_cart = len(user_df[user_df["behavior_type"] == 1])
        total_buy = len(user_df[user_df["behavior_type"] == 2])
        if total_buy > 0:
            return "intent"
        elif total_cart > 0:
            return "exploring"
        elif total_pv > 0:
            return "latent"
        else:
            return "dormant"

    def rerank_items(self, raw_rec_list: list, user_state: str) -> list:
        if user_state not in self.valid_state or not raw_rec_list:
            return raw_rec_list[:10]
        strategy = self.strategy_map[user_state]
        score_dict = {}
        for item in raw_rec_list:
            base_score = 1.0
            if user_state == "latent":
                base_score *= strategy["popular"]
            elif user_state == "exploring":
                base_score *= strategy["similar"]
            elif user_state == "intent":
                base_score *= strategy["click_seq"]
            elif user_state == "dormant":
                base_score *= strategy["recall"]
            score_dict[item] = base_score
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:10]]

def get_rerank_result(rec_list: list, user_df: pd.DataFrame) -> tuple[str, list]:
    reranker = DemandRerank()
    state = reranker.judge_user_state(user_df)
    res_list = reranker.rerank_items(rec_list, state)
    return state, res_list

if __name__ == "__main__":
    df_all = pd.read_csv(CHAINED_CSV_PATH)
    print(f"Loaded behavior data, total records: {len(df_all)}")

    test_uid = df_all["user_id"].unique()[0]
    user_data = df_all[df_all["user_id"] == test_uid]
    print(f"Selected test user ID: {test_uid}")
    print(f"Behavior records of this user: {len(user_data)}")

    test_rec_list = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011]
    user_state, rerank_result = get_rerank_result(test_rec_list, user_data)

    print(f"User demand state: {user_state}")
    print(f"Top10 items after reranking: {rerank_result}")