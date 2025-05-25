from transformers import (Trainer
)
import torch.nn as nn
import os
import shutil

class ExtractiveQATrainer(Trainer):
    """
    Custom trainer class for extractive question answering tasks.
    
    This trainer extends the base Trainer class to handle extractive QA tasks,
    which require special handling of answer spans and position tracking.
    
    Attributes:
        Inherits all attributes from the base Trainer class.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the loss for extractive question answering.
        
        This method handles the special case of extractive QA where we need to:
        1. Track start and end positions of answers
        2. Handle cases where answers are not found in the context
        
        Args:
            model: The model to compute loss for
            inputs (dict): Input tensors including input_ids, attention_mask, etc.
            return_outputs (bool, optional): Whether to return model outputs. Defaults to False.
            
        Returns:
            tuple or torch.Tensor: If return_outputs is True, returns (loss, outputs)
                                 Otherwise returns just the loss
        """
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
    """
    Custom trainer class that saves the model with the lowest training loss.
    
    This trainer extends the base Trainer class to implement a custom save strategy
    that keeps track of the best model based on training loss rather than
    validation metrics.
    
    Attributes:
        Inherits all attributes from the base Trainer class.
        best_loss (float): The best (lowest) training loss seen so far
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the trainer with best loss tracking.
        
        Args:
            *args: Variable length argument list passed to parent class
            **kwargs: Arbitrary keyword arguments passed to parent class
        """
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

def debug_print(title: str, **kwargs):
    """
    Print debug information in a formatted way.
    
    This utility function prints debug information with a consistent format,
    making it easier to track training progress and debug issues.
    
    Args:
        title (str): The title/description of the value being printed
        **kwargs: The values to print
    """
    lines = [f"- {name}: {value}" for name, value in kwargs.items()]
    max_line_length = max(len(line) for line in lines) if lines else 0

    # Fixed number of '=' on both sides of the title
    side_len = 10
    title_line = "=" * side_len + title + "=" * side_len

    # Total border length is 20 + len(title)
    border = "=" * (20 + len(title))

    print(title_line)
    for line in lines:
        print(line)
    print(border)
