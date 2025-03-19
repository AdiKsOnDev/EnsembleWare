import os
import glob
import logging
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

include_logger = logging.getLogger('include')

def fine_tune(model, training_data, testing_data, results_dir="./results/"):
    include_logger.debug(f"Fine tuning {model.model_name}")
    include_logger.debug(f"Model will be saved in {results_dir}")

    training_args = TrainingArguments(
        output_dir=f"{results_dir}{model.model_name}/",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_dir=f"{results_dir}{model.model_name}/logs",
        save_steps=300,
        gradient_checkpointing=True,
        logging_steps=10,
        save_total_limit=1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        resume_from_checkpoint=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        fp16=True,
        disable_tqdm=False,
        report_to="none",
        eval_strategy="epoch"
    )
    include_logger.debug("Training Args Created")

    data_collator = DataCollatorWithPadding(model.tokenizer, padding='longest')

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=testing_data,
        data_collator=data_collator,
    )

    include_logger.debug("Trainer object created")
    results_dir = f"{results_dir}/{model.model_name}/"
    checkpoints = glob.glob(os.path.join(results_dir, "checkpoint-*"))

    trainer.train(max(checkpoints, key=lambda x: int(x.split('-')[-1])))

    include_logger.debug("Training Successful")

    model.model.save_pretrained(
        f"{results_dir}/fine_tuned_{model.model_name}")
    model.tokenizer.save_pretrained(
        f"{results_dir}/fine_tuned_{model.model_name}")

