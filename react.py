# from transformers import CodeAgent, HfEngine

# llm_engine = HfEngine(model="Qwen/CodeQwen1.5-7B-Chat")
# agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)

# agent.run(
#     "Could you translate this sentence from French, say it out loud and return the audio.",
#     sentence="Où est la boulangerie la plus proche?",
# )
def llm_engine(messages, stop_sequences=["Task"])-> str:
    import openai
    openai.api_key = 
    openai.base_url = 
    completion = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages,
        stop=stop_sequences
    )
    return completion.choices[0].message.content

prompt = """
###code：
import pandas as pd
data = {"header": ["School", "Location", "Outright Titles", "Shared Titles", "Runners-Up", "Total Finals", "Last Title", "Last Final"], "rows": [["Methodist College Belfast", "Belfast", "35", "2", "25", "62", "2014", "2014"], ["Royal Belfast Academical Institution", "Belfast", "29", "4", "21", "54", "2007", "2013"], ["Campbell College", "Belfast", "23", "4", "12", "39", "2011", "2011"], ["Coleraine Academical Institution", "Coleraine", "9", "0", "24", "33", "1992", "1998"], ["The Royal School, Armagh", "Armagh", "9", "0", "3", "12", "2004", "2004"], ["Portora Royal School", "Enniskillen", "6", "1", "5", "12", "1942", "1942"], ["Bangor Grammar School", "Bangor", "5", "0", "4", "9", "1988", "1995"], ["Ballymena Academy", "Ballymena", "3", "0", "6", "9", "2010", "2010"], ["Rainey Endowed School", "Magherafelt", "2", "1", "2", "5", "1982", "1982"], ["Foyle College", "Londonderry", "2", "0", "4", "6", "1915", "1915"], ["Belfast Royal Academy", "Belfast", "1", "3", "5", "9", "1997", "2010"], ["Regent House Grammar School", "Newtownards", "1", "1", "2", "4", "1996", "2008"], ["Royal School Dungannon", "Dungannon", "1", "0", "4", "5", "1907", "1975"], ["Annadale Grammar School (now Wellington College)", "Belfast", "1", "0", "1", "2", "1958", "1978"], ["Ballyclare High School", "Ballyclare", "1", "0", "1", "2", "1973", "2012"], ["Belfast Boys' Model School", "Belfast", "1", "0", "0", "1", "1971", "1971"], ["Grosvenor High School", "Belfast", "1", "0", "0", "1", "1983", "1983"], ["Wallace High School", "Lisburn", "0", "0", "4", "4", "N/A", "2007"], ["Derry Academy", "Derry", "0", "0", "2", "2", "N/A", "1896"], ["Dalriada School", "Ballymoney", "0", "0", "1", "1", "N/A", "1993"], ["Galway Grammar School", "Galway", "0", "0", "1", "1", "N/A", "1887"], ["Lurgan College", "Lurgan", "0", "0", "1", "1", "N/A", "1934"], ["Omagh Academy", "Omagh", "0", "0", "1", "1", "N/A", "1985"], ["Sullivan Upper School", "Holywood", "0", "0", "1", "1", "N/A", "2014"]]}
# 转换为 Pandas DataFrame
df = pd.DataFrame(data['rows'], columns=data['header'])
###instrction:For question “"what is the difference in runners-up from coleraine academical institution and royal school dungannon?”，Please help me follow the above code to complete the answer to the question.the above code must be completely saved in the code block.
"""
from transformers.agents import ReactCodeAgent, PythonInterpreterTool

python_interpreter = PythonInterpreterTool()
agent = ReactCodeAgent(tools=[])
# import ipdb
# ipdb.set_trace()
agent.run(prompt)