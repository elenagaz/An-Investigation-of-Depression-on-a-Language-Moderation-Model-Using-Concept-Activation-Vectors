from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

import json

LABELS_NUM = {0: 'H', 1: 'H2', 2: 'HR', 3: 'OK', 4: 'S', 5: 'S3', 6: 'SH', 7: 'V', 8: 'V2'}


class ModerationDataset(lit_dataset.Dataset):
    LABELS = ['H', 'H2', 'HR', 'OK', 'S', 'S3', 'SH', 'V', 'V2']

    def __init__(self, file_path):
        super().__init__()
        self._examples = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                example = {
                    "prompt": record["text"],
                    "label": LABELS_NUM[record["target"]]
                }
                self._examples.append(example)

    def spec(self) -> lit_types.Spec:
        return {
            "prompt": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS)
        }
