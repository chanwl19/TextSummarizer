import os
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import BartForConditionalGeneration, BartTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"]
)

model_name = "facebook/bart-large-xsum"
model = BartForConditionalGeneration.from_pretrained(model_name)
bart_tokenizer = BartTokenizer.from_pretrained(model_name)

# Define the route for summarization
@app.post("/summarize")
async def summarize(request: Request):

    body = await request.json()
    text = body.get('text')
    max_word = body.get('max', 250)

    #Get abstractive summary
    abs_summary = get_abstractive(text, max_word)
    ext_summary = get_extractive(text, max_word)

    abs_new_words = compare_paragraphs(text, abs_summary)
    ext_new_words = compare_paragraphs(text, ext_summary)

    return {"abs_summary": abs_summary, 'ext_summary': ext_summary, 'abs_new_words' : abs_new_words , 'ext_new_words': ext_new_words}

def get_abstractive(text: str, max_len: int) -> str:
    print('start generating abstractive summary...')
    inputs = bart_tokenizer.encode("summarize: " + text, max_length=1024, return_tensors="pt", truncation=True)
    print('finished tokenizing...')

    print('start to use BART model to generate abstractive summary...')
    summary_ids = model.generate(inputs, 
                                 max_length=max_len, 
                                 min_length=50,
                                 num_beams=8, 
                                 length_penalty=2.5,
                                 early_stopping=True)

    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print('finished generate abstractive summary...')
    return summary

def get_extractive(text: str, max_len: int) -> str:
    print('start generating extractive summary...')
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    
    print('start to generate summary sentences...')
    summary_sentences = summarizer(parser.document, 10) 
    summary = " ".join([str(sentence) for sentence in summary_sentences])
    
    summary_words = summary.split()
    if len(summary_words) > max_len:
        summary = " ".join(summary_words[:max_len   ])
    
    return summary

def compare_paragraphs(paragraph1:str, paragraph2:str) -> list[str]:

    words1 = set(re.findall(r'\b\w+\b', paragraph1.lower()))
    words2 = set(re.findall(r'\b\w+\b', paragraph2.lower()))
    
    diff2 = words2 - words1
    
    return list(diff2)
