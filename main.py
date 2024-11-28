import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset

# Create FastAPI app
app = FastAPI()

model_path = "./model"
model_name = "facebook/bart-large-xsum"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def check_model_exists():
    #print(os.path.exists(model_path))
    return True

def train_model():

    print('Start to train model....')
    dataset = load_dataset("xsum", trust_remote_code=True)
    processed_dataset = dataset.map(preprocess_data, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,  # Directory to save the model
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,  # Enables summary generation during evaluation
        logging_dir="./logs",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    result = trainer.evaluation()
    print('evaluate result ' + result)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    is_trained = True
    print('Finish to train model....')

def preprocess_data(batch):
    inputs = tokenizer(
        batch["document"], 
        max_length=1024, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    targets = tokenizer(
        batch["summary"], 
        max_length=128, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": targets["input_ids"]}

if not check_model_exists():
    train_model()
else :
    print('Model trained')

# Define the route for summarization
@app.post("/summarize")
async def summarize(request: Request):
    body = await request.json()
    text = body.get('text')
    max_word = body.get('max', 250)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    if not check_model_exists(): # should be true
        model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)

    print('start summarizing...')
    # Tokenize input text and generate summary
    inputs = tokenizer.encode("summarize: " + text, max_length=1024, return_tensors="pt", truncation=True)
    print('finished tokenizing...')
    # Generate summary using BART
    summary_ids = model.generate(inputs, 
                                 max_length=max_word, 
                                 min_length=50,
                                 num_beams=8, 
                                 length_penalty=2.5,
                                 early_stopping=True)

    
    print('finished generate...')
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print('finished summary... ' + summary)
    return {"summary": summary}

