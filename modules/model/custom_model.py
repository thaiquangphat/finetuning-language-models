from transformers import (
    GPT2LMHeadModel # For GPT-2
)
import torch.nn as nn

class GPT2ForExtractiveQA(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Adding custom linear layers for start and end positions
        self.start_classifier = nn.Linear(config.n_embd, 1)
        self.end_classifier = nn.Linear(config.n_embd, 1)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, start_positions=None, end_positions=None):
        # Get the GPT2 outputs
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

        # Predict start and end positions using linear layers
        start_logits = self.start_classifier(hidden_states).squeeze(-1)  # Shape: (batch_size, seq_len)
        end_logits = self.end_classifier(hidden_states).squeeze(-1)    # Shape: (batch_size, seq_len)

        # If labels are provided (for training), calculate the loss
        if labels is not None:
            # Compute the loss (using CrossEntropyLoss)
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            return loss, (start_logits, end_logits)
        
        return start_logits, end_logits