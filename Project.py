from transformers import T5ForConditionalGeneration, T5Tokenizer
import streamlit as at
from streamlit_pdf_viewer import pdf_viewer
from rouge_score import rouge_scorer
from PyPDF2 import PdfFileReader
import io



#T5 Initialization
model_name = "t5-Base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy = False)

#Translations
def TranslationGerman(summary):
    input_ids = tokenizer("translate to German: " + summary, return_tensors="pt", max_length=512, truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=100, min_length=25)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

def TranslationRomanian(summary):
    input_ids = tokenizer("translate to Romanian: " + summary, return_tensors="pt",max_length=512, truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=100, min_length=25)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

def TranslationFrench(summary):
    input_ids = tokenizer("translate to French: " + summary, return_tensors="pt",max_length=512, truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=100, min_length=25)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

#Summarizing the text
def summarize(texto, model, tokenizer):
    input_ids = tokenizer.encode(texto, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=100, min_length=25, length_penalty=2.0, num_beams=4,no_repeat_ngram_size=3)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

#Extracting PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfFileReader(pdf_file)
    pdf = ""
    for page_num in range(pdf_reader.getNumPages()):
        pages = pdf_reader.getPage(page_num)
        pdf += pages.extract_text()
    return pdf

#using paraphrase function for t5 
def paraPhrase(summary):
    inputs = tokenizer.encode(summary, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=55, num_beams=4, early_stopping=True)
    reference_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reference_summary

def calculate_rouge(reference_summary, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, summary)
    return scores



#Ui 
at.title("T5 Summarization and Translation")
texto = at.text_area("Enter Text to be Summarized and/or Translated", height=350)
languages = ["German", "French", "English","Romanian"]
select_languages = at.selectbox("Select a language: ", languages)
uploaded_pdf = at.file_uploader("Click for a PDF File",type='pdf')
documentRead = ""
if uploaded_pdf is not None:
            document = uploaded_pdf.read()
            pdf_file = io.BytesIO(document)
            documentRead = extract_text_from_pdf(pdf_file)
            pdf_viewer(document)


#multiple buttons for streamlit
col1, col2 , col3 = at.columns(3)
def stateful_button(*args, key=None, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in at.session_state:
        at.session_state[key] = False

    if at.button(*args, **kwargs):
        at.session_state[key] = not at.session_state[key]
    return at.session_state[key]

with col1:
    if stateful_button('Summarize', key="Summarize"):
        if texto:
           at.write("Summary of Text")
           summary = summarize(texto,model,tokenizer)
           at.write(summary)
        if documentRead:
            at.write("Summary of PDF")
            summary2 = summarize(documentRead,model,tokenizer)
            at.write(summary2)
    else:
        at.write("Enter Text or Upload PDF")

with col2:
    if stateful_button("Translate", key="Translation"):
        summary = summarize(texto,model,tokenizer)
        summary2 = summarize(documentRead,model,tokenizer)
        if texto:
            at.write("Text Translation")
            if select_languages == "German":
               translation = TranslationGerman(summary)
               at.write(translation)
            elif select_languages == "English":
               at.write(summary)
            elif select_languages == "French":
               translation = TranslationFrench(summary)
               at.write(translation)
            elif select_languages == "Romanian":
              translation = TranslationRomanian(summary)
              at.write(translation)
        if uploaded_pdf :
            at.write("PDF Translation")
            if select_languages == "German":
              translation = TranslationGerman(summary2)
              at.write(translation)
            elif select_languages == "English":
              at.write(summary2)
            elif select_languages == "French":
              translation = TranslationFrench(summary2)
              at.write(translation)
            elif select_languages == "Romanian":
              translation = TranslationRomanian(summary2)
              at.write(translation)
    else:
        at.write("Please Select a language.")
        
      
with col3:
    if stateful_button('Rouge Score', key="Score"):
        if texto:
         at.write("Rouge Score for Text")
         reference_summary = paraPhrase(summary)
         rouge_output = calculate_rouge(summary,reference_summary)
         at.write("ROUGE-1: ", rouge_output['rouge1'])
         at.write("ROUGE-2: ", rouge_output['rouge2'])
         at.write("ROUGE-L: ", rouge_output['rougeL'])
        if documentRead:
         at.write("Rouge Score for PDF")
         reference_summary = paraPhrase(summary2)
         rouge_output = calculate_rouge(summary2,reference_summary)
         at.write("ROUGE-1: ", rouge_output['rouge1'])
         at.write("ROUGE-2: ", rouge_output['rouge2'])
         at.write("ROUGE-L: ", rouge_output['rougeL'])
         
    else :
        at.write("Please Create a summary to scored")


    #To Run using streamlit use this on the terminal 
#streamlit run  "/Users/juliorivera/Library/Mobile Documents/com~apple~CloudDocs/MERCER/ECE 691/FinalProject/Project.py"
