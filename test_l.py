from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import ipdb
import csv
import jsonlines
import evaluation_lla

path = "/root/mutil_agent/FREB-TQA/dataset/02_rel/wtq_nonfactoid_dev.json"
def reformat_table(table):
    if isinstance(table, list):
        header = table[0]
    elif isinstance(table, dict):
        header = table['header']
    output = ' | '.join(header)
    output += '\n'
    if isinstance(table, list):
        rows = table[1:] 
    elif isinstance(table, dict):
        rows = table['rows']
    for row in rows:
        output += ' | '.join(row)
        output += '\n'
    return output

def Response(model, tokenizer,ids, query,answer, table):
    time = 0
    results = []
    for i, (id, q,a, t) in tqdm(enumerate(zip(ids, query,answer, table))):

        time += 1
        if time == 101:
            break
        table_string = reformat_table(t)
        prompt = f"Answer the question according to the table and. Return the answer following Answer[final answer]. Table: {table_string} Question:{q}\nAnswer:"
        

        # 对提示进行编码
        inputs = tokenizer(prompt, return_tensors="pt")

        # 生成文本
        generate_ids = model.generate(
        inputs["input_ids"], 
        max_length=15000, 
        num_return_sequences=1,  # 默认值就是1，这里只是为了清晰
        temperature=0.1,         # 控制生成文本的多样性
        )

        # 解码生成的文本
        decoded_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # decoded_string = ''.join(decoded_output)


        # answer_start_index = decoded_string.find('Answer:')
        # answer_start = decoded_output.index('Answer') + len('Answer')
        # answer = decoded_output[answer_start:].strip()
        answer_start_index = decoded_output.find('Answer:')
        answer = decoded_output[answer_start_index:]

        print("Generated Response:")
        print(answer)
        results.append({
                'number' : i,
                'id' : id,
                'query': q,
                'answer' : a,
                'table': t,
                'llm_answer' : answer
        })
    with jsonlines.open('./lla.jsonl', mode='w') as writer:
        for result in results:
            writer.write(result)
        # with open('./lla.csv', 'a') as f:
        #     csv_writer = csv.DictWriter(f, field)
        #     csv_writer.writerow({
        #         'number' : i,
        #         'id' : id,
        #         'query': q,
        #         'answer' : a,
        #         'table': t,
        #         'llm_answer' : answer
        #     })
if __name__ == "__main__":
    # 加载预训练的模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("cspencergo/llama-2-7b-tabular")
    model = AutoModelForCausalLM.from_pretrained("cspencergo/llama-2-7b-tabular")

    data = [
    ]
    
    field = ['number',"ids", 'query', 'answer', 'table', 'llm_answer']
    # with open('./lla.csv', 'w') as cs:
    #     csv_writer = csv.DictWriter(cs, field)
    #     csv_writer.writeheader()
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    ids = [item['id'] for item in data]
    query = [item['question'] for item in data]
    answer = [item['answers'] for item in data]
    table = [item['table'] for item in data]
  

    Response(model, tokenizer=tokenizer,ids=ids,query=query,answer = answer, table=table)
    evaluation_lla.main()
    

    
