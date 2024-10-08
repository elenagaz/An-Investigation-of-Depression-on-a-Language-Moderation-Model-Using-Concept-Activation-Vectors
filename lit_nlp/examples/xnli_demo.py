r"""Example demo for multilingual NLI on the XNLI eval set.

To run locally with our trained model:
  python -m lit_nlp.examples.xnli_demo --port=5432 #####--works

Then navigate to localhost:5432 to access the demo UI.

To train a model for this task, use tools/glue_trainer.py or your favorite
trainer script to fine-tune a multilingual encoder, such as
bert-base-multilingual-cased, on the mnli task.

Note: the LIT UI can handle around 10k examples comfortably, depending on your
hardware. The monolingual (english) eval sets for MNLI are about 9.8k each,
while each language for XNLI is about 2.5k examples, so we recommend using the
--languages flag to load only the languages you're interested in.
"""

from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.examples.datasets import classification
from lit_nlp.examples.datasets import glue
from lit_nlp.lib import file_cache
from lit_nlp.examples.glue2 import glue_models2


# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_LANGUAGES = flags.DEFINE_list(
    "languages", ["en", "es", "hi", "zh"],
    "Languages to load from XNLI. Available languages: "
    "ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,zh,vi"
)

_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    "https://storage.googleapis.com/what-if-tool-resources/lit-models/mbert_mnli.tar.gz",
    (
        "Path to fine-tuned model files. Expects model to be in standard "
        "transformers format, e.g. as saved by model.save_pretrained() and "
        "tokenizer.save_pretrained()."
    ),
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", None, "Maximum number of examples to load into LIT. "
    "Note: MNLI eval set is 10k examples, so will take a while to run and may "
    "be slow on older machines. Set --max_examples=200 for a quick start.")


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
  """Returns a LitApp instance for consumption by gunicorn."""
  FLAGS.set_default("server_type", "external")
  FLAGS.set_default("demo_mode", True)
  # Parse flags without calling app.run(main), to avoid conflict with
  # gunicorn command line flags.
  unused = flags.FLAGS(sys.argv, known_only=True)
  if unused:
    logging.info("xnli_demo:get_wsgi_app() called with unused args: %s", unused)
  return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Normally path is a directory; if it's an archive file, download and
  # extract to the transformers cache.
  model_path = _MODEL_PATH.value
  if model_path.endswith(".tar.gz"):
    model_path = file_cache.cached_path(
        model_path, extract_compressed_file=True)

  models = {"nli": glue_models2.MNLIModel(model_path, inference_batch_size=16)}
  datasets = {
      "xnli": classification.XNLIData("validation", _LANGUAGES.value),
      "mnli_dev": glue.MNLIData("validation_matched"),
      "mnli_dev_mm": glue.MNLIData("validation_mismatched"),
  }

  # Truncate datasets if --max_examples is set.
  for name in datasets:
    logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))
    datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
    logging.info("  truncated to %d examples", len(datasets[name]))

  server_options = server_flags.get_flags()
  server_options['port'] = 8080  # Change to 8080

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_options)
  return lit_demo.serve()


if __name__ == "__main__":
  app.run(main)
