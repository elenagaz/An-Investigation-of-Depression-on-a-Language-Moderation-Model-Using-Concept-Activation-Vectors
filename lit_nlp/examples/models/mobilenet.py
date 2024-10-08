"""MobileNet model trained on ImageNet dataset."""

from lit_nlp.api import model
from lit_nlp.api import types as lit_types
from lit_nlp.examples.models import imagenet_labels
from lit_nlp.lib import image_utils
import numpy as np
import tensorflow as tf

# Internal shape of the model input (h, w, c).
IMAGE_SHAPE = (224, 224, 3)


class MobileNet(model.BatchedModel):
  """MobileNet model trained on ImageNet dataset."""

  def __init__(self, name='mobilenet_v2') -> None:
    # Initialize imagenet labels.
    self.labels = [''] * len(imagenet_labels.IMAGENET_2012_LABELS)
    self.label_to_idx = {}
    for i, l in imagenet_labels.IMAGENET_2012_LABELS.items():
      l = l.split(',', 1)[0]
      self.labels[i] = l
      self.label_to_idx[l] = i
      #self.labels = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich']
      # self.label_to_idx = {
      # 'tench': 0,
      # 'goldfish': 1,
      # 'great white shark': 2,
      # 'tiger shark': 3,
      # 'hammerhead': 4,
      # 'electric ray': 5,
      # 'stingray': 6,
      # 'cock': 7,
      # 'hen': 8,
      # 'ostrich': 9
      # }

    if name == 'mobilenet_v2':
      self.model = tf.keras.applications.mobilenet_v2.MobileNetV2()
    elif name == 'mobilenet':
      self.model = tf.keras.applications.mobilenet.MobileNet()

  def predict_minibatch(
      self, input_batch: list[lit_types.JsonDict]
  ) -> list[lit_types.JsonDict]:
    output = []
    for example in input_batch:
      # Convert input to the model acceptable format.
      img_data = example['image']
      if isinstance(img_data, str):
        img_data = image_utils.convert_image_str_to_array(img_data, IMAGE_SHAPE)
      # Get predictions.
      x = img_data[np.newaxis, ...]
      x = tf.convert_to_tensor(x)
      preds = self.model(x).numpy()[0]
      # Determine the gradient target.
      if (grad_target := example.get('label')) is None:
        grad_target_idx = np.argmax(preds)
      else:
        grad_target_idx = self.label_to_idx[grad_target]
      # Calculate gradients.
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = self.model(x)[0, grad_target_idx]
        grads = tape.gradient(y, x).numpy()[0]
      # Add results to the output.
      output.append({
          'preds': preds,
          'grads': grads,
      })

    return output

  def input_spec(self):
    return {
        'image': lit_types.ImageBytes(),
        # If `grad_target` is not specified then the label with the highest
        # predicted score is used as the gradient target.
        'label': lit_types.CategoryLabel(vocab=self.labels, required=False),
    }

  def output_spec(self):
    return {
        'preds': lit_types.MulticlassPreds(
            vocab=self.labels, autosort=True, parent='label'
        ),
        'grads': lit_types.ImageGradients(
            align='image', grad_target_field_key='label'
        ),
    }
