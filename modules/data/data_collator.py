from transformers import DataCollatorForSeq2Seq
import numpy as np

class FastDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Manually fix labels first
        for f in features:
            if isinstance(f.get('labels'), np.ndarray):
                f['labels'] = f['labels'].tolist()
                
        batch = super().__call__(features)
        return batch