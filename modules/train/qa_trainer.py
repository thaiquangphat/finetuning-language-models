from transformers import (Trainer
)
import torch.nn as nn

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
