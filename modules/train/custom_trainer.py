from transformers import (Trainer
)
import torch.nn as nn
import os
import shutil

class ExtractiveQATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass through the model
        # print("Compute loss in QA Extractive")
        # print(inputs.keys())
        start_logits, end_logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            start_positions=inputs["start_positions"],
            end_positions=inputs["end_positions"]
        )

        # Get the true start and end positions from the inputs
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]

        # Use CrossEntropy loss for both start and end positions
        loss_fct = nn.CrossEntropyLoss()

        # Compute the loss for start and end positions
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)

        # Combine the two losses
        loss = (start_loss + end_loss) / 2

        # Return the loss and the outputs if needed
        return (loss, (start_logits, end_logits)) if return_outputs else loss

class LeastTrainLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_train_loss = float("inf")
        self.best_model_path = None

    def _save_checkpoint(self, model, trial):
        current_train_loss = self.state.log_history[-1].get("loss", float("inf"))
        if current_train_loss < self.best_train_loss:
            self.best_train_loss = current_train_loss
            output_dir = os.path.join(self.args.output_dir, "best_model")
            self.save_model(output_dir)
            self.best_model_path = output_dir
            print(f"New best model saved with training loss: {self.best_train_loss}")
        super()._save_checkpoint(model, trial)

    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)
        if self.best_model_path:
            final_path = os.path.join(self.args.output_dir, "final_best_model")
            shutil.copytree(self.best_model_path, final_path, dirs_exist_ok=True)
            print(f"Best model re-saved after training to: {final_path}")
        return output

