import logging
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

logger = logging.getLogger('include')

def fine_tune(model, training_data, testing_data, results_dir="./results/"):
    logger.debug(f"Fine tuning {model.model_name}")
    logger.debug(f"Model will be saved in {results_dir}")

    training_args = TrainingArguments(
        output_dir=f"{results_dir}{model.model_name}/",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir=f"{results_dir}{model.model_name}/logs",
        save_steps=1000,
        gradient_checkpointing=True,
        logging_steps=10,
        save_total_limit=1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        fp16=True,
        disable_tqdm=False,
        report_to="none",
        eval_strategy="epoch"
    )
    logger.debug("Training Args Created")

    data_collator = DataCollatorWithPadding(model.tokenizer, padding='longest')

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=testing_data,
        data_collator=data_collator,
    )

    logger.debug("Trainer object created")
    
    trainer.train()

    logger.debug("Training Successful")

    model.model.save_pretrained(
        f"{results_dir}/{model.model_name}/fine_tuned_{model.model_name}")
    model.tokenizer.save_pretrained(
        f"{results_dir}/{model.model_name}/fine_tuned_{model.model_name}")

