import sys
import os
from collections.abc import Sequence
from typing import Optional

from lit_nlp import dev_server
from lit_nlp import server_flags
from absl import flags
from absl import logging
from lit_nlp.my_model_moderation.model_validation import moderation_tf
from lit_nlp.my_model_moderation.model_validation import moderation_dataset_validation

from absl import app

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    "KoalaAI/Text-Moderation",
    "Path to saved model (from transformers library).",
)

script_dir = os.path.dirname(os.path.abspath(__file__))

relative_file_path = os.path.join('KoalaAI_Text-Moderation-v2-small', 'converted_valid_data-00000-of-00001.jsonl')
file_path = os.path.join(script_dir, relative_file_path)


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
    """Returns a LitApp instance """
    FLAGS.set_default("server_type", "external")
    FLAGS.set_default("demo_mode", True)
    # Parse flags without calling app.run(main), to avoid conflict with
    # gunicorn command line flags.
    unused = flags.FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info(
            "moderation_demo:get_wsgi_app() called with unused args: %s", unused
        )
    return main([])


# main
def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    model_path = _MODEL_PATH.value
    logging.info("Working directory: %s", model_path)

    # Load our trained model.
    model = {"moderation": moderation_tf.ModerationModel(model_path)}
    datasets = {"moderation_dataset": moderation_dataset_validation.ModerationDataset(file_path=file_path)}

    server_options = server_flags.get_flags()
    server_options['port'] = 8080

    lit_demo = dev_server.Server(model, datasets, **server_options)
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
