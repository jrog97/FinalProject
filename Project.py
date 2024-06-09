from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
import gradio as gr

#Goal of the Project is the user can choose between a summary or translation of both with German as the translation. if I have time we can do another language

#Using T5 to create the summarization and translation

#Here is the basis for the summarization



model_name = 'T5-Base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy = False,trust_remote_code=True,force_download=True)
text =("""What are felons barred from doing? It varies by state. In New York, where Trump was convicted, there are collateral consequences of being 
convicted of a felony. Importantly, felons in New York cannot hold many public offices, including elected positions. But Trump is no longer a New York resident. """)


inputs = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
summary_ids = model.generate(inputs, max_length=50, min_length=25, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\nSummary:" + summary + "\n")

#here is for the translation
input_ids = tokenizer("translate English to Spanish: " + summary, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=25)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
translation = ("Translation: " + decoded + "\n" )
print(translation)