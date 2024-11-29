from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_path = "./model"
model_name = "facebook/bart-large-xsum"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

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
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": targets["input_ids"]} """


