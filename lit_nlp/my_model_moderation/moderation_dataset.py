from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

import json


class ModerationDataset(lit_dataset.Dataset):
    LABELS = ['H', 'H2', 'HR', 'OK', 'S', 'S3', 'SH', 'V', 'V2']

    def __init__(self, file_path):
        super().__init__()
        self._examples = []

        with open(file_path, 'r') as file:
            for line in file:
                record = json.loads(line)
                self._examples.append({
                    "prompt": record["prompt"],
                    "label": record['label'],
                })

    def spec(self) -> lit_types.Spec:
        return {
            "prompt": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS)
        }
