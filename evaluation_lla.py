
from evaluator import to_value_list, check_denotation
import json

def exact_match_score(pred_path, gold_path):
    """
    calculate em 
    """
    results = {'non-truncate': 0, 'all': 0}
    with open(pred_path, "r") as f:
        pred = [json.loads(line) for line in f]
    with open(gold_path, "r") as f:
        factoid = [json.loads(line) for line in f]
        answers = {item['id']: item['answers'] for item in factoid}
    # align ids
    gold = [answers[item["id"]] for item in pred]
    for p, g in zip(pred, gold):
        p_ = p["llm_answer"]
        if not isinstance(p_, list):
            p_ = [p_]
        if not isinstance(g, list):
            g = [g]
        if check_denotation(to_value_list(g), to_value_list(p_)):
            if "truncate" in p.keys() and not p["truncate"]:
                results["non-truncate"] += 1
            results["all"] += 1
    if "truncate" in pred[0].keys():
        non_truncate_total = len(
            [item for item in pred if not item["truncate"]])
        return results["all"]/len(pred), results["non-truncate"]/non_truncate_total
    return results["all"]/len(pred), 0

def main():
    pred_path = "/root/mutil_agent/LLAMA_TABLE/lla.jsonl"
    gold_path = "/root/mutil_agent/LLAMA_TABLE/test.jsonl"
    acculation , b = exact_match_score(pred_path, gold_path)
    accuracy_percentage = round(acculation * 100, 2)
    print(f"准确率: {accuracy_percentage}%")
if __name__ == "__main__":
    main()