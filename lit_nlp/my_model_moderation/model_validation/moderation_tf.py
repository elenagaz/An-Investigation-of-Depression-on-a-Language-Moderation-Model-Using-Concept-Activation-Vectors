"""Wrapper for moderation model and moderation model class - able to use in LIT
"""
import logging
import os
# from abc import ABC
from typing import Optional, Sequence, Any, Iterable

import numpy as np
import torch
import transformers
from lit_nlp.api import types as lit_types
from lit_nlp.api import model as lit_model
import threading

import attr
import re
import tensorflow as tf
from lit_nlp.examples.models import model_utils

from lit_nlp.lib import utils

JsonDict = lit_types.JsonDict
Spec = lit_types.Spec
TFSequenceClassifierOutput = (
    transformers.modeling_tf_outputs.TFSequenceClassifierOutput
)


@attr.s(auto_attribs=True, kw_only=True)
class ModerationConfig(object):
    prompt: str = "prompt"
    label_names: str = "label"

    max_seq_length: int = 528
    inference_batch_size: int = 10  # Over 10 Examples results in Memory being exhausted

    # change is permanent
    labels: Optional[list[str]] = ['H', 'H2', 'HR', 'OK', 'S', 'S3', 'SH', 'V', 'V2']
    null_label_idx: Optional[int] = None
    compute_grads: bool = True
    output_attention: bool = True
    output_embeddings: bool = True

    @classmethod
    def init_spec(cls) -> lit_types.Spec:
        return {
            # probably not needed
            "max_seq_length": lit_types.Integer(
                default=128,
                max_val=528,
                min_val=1,
                required=False,
            ),
            # removed others
            "compute_grads": lit_types.Boolean(default=True, required=False),
            "output_attention": lit_types.Boolean(default=True, required=False),
            "output_embeddings": lit_types.Boolean(default=True, required=False),
        }


class ModerationModel(lit_model.BatchedModel):

    def __init__(self, model_name, **config_kw):
        self.config = ModerationConfig(**config_kw)
        self._load_model(model_name)
        self._lock = threading.Lock()

    def _verify_num_layers(self, hidden_states: Sequence[Any]):
        """Verify correct # of layer activations returned."""
        # First entry is embeddings, then output from each transformer layer.
        expected_hidden_states_len = self.model.config.num_hidden_layers + 1
        actual_hidden_states_len = len(hidden_states)
        if actual_hidden_states_len != expected_hidden_states_len:
            raise ValueError(
                "Unexpected size of hidden_states. Should be one "
                "more than the number of hidden layers to account "
                "for the embeddings. Expected "
                f"{expected_hidden_states_len}, got "
                f"{actual_hidden_states_len}."
            )

    @property
    def is_regression(self) -> bool:
        return self.config.labels is None

    def _load_model(self, model_name):
        """Load model. Can be overridden for testing."""
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.vocab = self.tokenizer.convert_ids_to_tokens(
            range(len(self.tokenizer)))
        model_config = transformers.AutoConfig.from_pretrained(model_name)
        self.model = model_utils.load_pretrained(
            transformers.TFAutoModelForSequenceClassification,
            model_name,
            config=model_config,
            from_pt=True)
        # summary = self.model.summary()

    def _get_tokens(self, ex: JsonDict, field_name: str) -> list[str]:
        with self._lock:
            return (ex.get("tokens_" + field_name) or
                    self.tokenizer.tokenize(ex[field_name]))

    def _preprocess(self, inputs: Iterable[JsonDict]) -> dict[str, tf.Tensor]:
        # Use pretokenized input if available.
        tokens_prompt = [self._get_tokens(ex, self.config.prompt) for ex in inputs]
        # looks like this ['i', 'Ġhate', 'Ġit', 'Ġhere']

        # Use custom tokenizer call to make sure we don't mangle pre-split
        # wordpieces in pretokenized input.
        encoded_input = model_utils.batch_encode_pretokenized(
            self.tokenizer,
            tokens_prompt,
            max_length=self.config.max_seq_length)
        return encoded_input  # pytype: disable=bad-return-type #Is from glue models

    def _make_dataset(self, inputs: Iterable[JsonDict]) -> tf.data.Dataset:
        """Make a tf.data.Dataset from inputs in LIT format."""
        encoded_input = self._preprocess(inputs)

        if self.is_regression:
            labels = tf.constant([ex[self.config.label_names] for ex in inputs],
                                 dtype=tf.float32)
        else:
            labels = tf.constant([
                self.config.labels.index(ex[self.config.label_names]) for ex in inputs
            ],
                dtype=tf.int64)
        # encoded_input is actually a transformers.BatchEncoding
        # object, which tf.data.Dataset doesn't like. Convert to a regular dict.
        return tf.data.Dataset.from_tensor_slices((dict(encoded_input), labels))

    def save(self, path: str):
        """Save model weights and tokenizer info.

        To re-load, pass the path to the constructor instead of the name of a
        base model.

        Args:
          path: directory to save to. Will write several files here.
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    def _segment_slicers(self, tokens: list[str]):
        """Slicers along the tokens dimension for each segment.

        For tokens ['[CLS]', a0, a1, ..., '[SEP]', b0, b1, ..., '[SEP]'], #spec for bert classification & seperator
        we want to get the slices [a0, a1, ...] and [b0, b1, ...]

        Args:
          tokens: <string>[num_tokens], including special tokens

        Returns:
          (slicer_a, slicer_b), slice objects
        """
        try:
            split_point = tokens.index(self.tokenizer.sep_token)
        except ValueError:
            split_point = len(tokens) - 1
        slicer_a = slice(1, split_point)  # start after [CLS]
        slicer_b = slice(split_point + 1, len(tokens) - 1)  # end before last [SEP]
        return slicer_a, slicer_b

    def _postprocess(self, output: dict[str, Any]):
        """Per-example postprocessing, on NumPy output."""
        ntok = output.pop("ntok")
        output["tokens"] = self.tokenizer.convert_ids_to_tokens(output.pop("input_ids")[:ntok])

        # Tokens for each segment, individually.
        slicer_a, slicer_b = self._segment_slicers(output["tokens"])
        output["tokens_" + self.config.prompt] = output["tokens"][slicer_a]

        # Embeddings for each segment, individually.
        if self.config.output_embeddings:
            output["input_embs_" + self.config.prompt] = (output["input_embs"][slicer_a])

        # Gradients for each segment, individually.
        if self.config.compute_grads:
            # Gradients for the CLS token.
            output["cls_grad"] = output["input_emb_grad"][0]
            output["token_grad_" +
                   self.config.prompt] = output["input_emb_grad"][slicer_a]

            # TODO(b/294613507): remove output[self.config.label_name] once TCAV
            # is updated.
            if not self.is_regression:
                # Return the label corresponding to the class index used for gradients.
                output[self.config.label_names] = self.config.labels[
                    output[self.config.label_names]
                ]  # pytype: disable=container-type-mismatch

            # Remove "input_emb_grad" since it's not in the output spec.
            del output["input_emb_grad"]

        if not self.config.output_attention:
            return output

        # Process attention.
        for key in output:
            if not re.match(r"layer_(\d+)/attention", key):
                continue
            # Select only real tokens, since most of this matrix is padding.
            # <float32>[num_heads, max_seq_length, max_seq_length]
            # -> <float32>[num_heads, num_tokens, num_tokens]
            output[key] = output[key][:, :ntok, :ntok].transpose((0, 2, 1))
            # Make a copy of this array to avoid memory leaks, since NumPy otherwise
            # keeps a pointer around that prevents the source array from being GCed.
            output[key] = output[key].copy()  # pytype: disable=attribute-error

        return output

    # copied + changes in name too
    def _scatter_embs(self, passed_input_embs, input_embs, batch_indices, offsets):
        """Scatters custom passed embeddings into the default model embeddings.

        Args:
          passed_input_embs: <tf.float32>[num_scatter_tokens], the custom passed
            embeddings to be scattered into the default model embeddings.
          input_embs: the default model embeddings.
          batch_indices: the indices of the embeddings to replace in the format
            (batch_index, sequence_index).
          offsets: the offset from which to scatter the custom embedding (number of
            tokens from the start of the sequence).

        Returns:
          The default model embeddings with scattered custom embeddings.
        """

        # <float32>[scatter_batch_size, num_tokens, emb_size]
        filtered_embs = [emb for emb in passed_input_embs if emb is not None]

        # Prepares update values that should be scattered in, i.e. one for each
        # of the (scatter_batch_size * num_tokens) word embeddings.
        # <np.float32>[scatter_batch_size * num_tokens, emb_size]
        updates = np.concatenate(filtered_embs)

        # Prepares indices in format (batch_index, sequence_index) for all
        # values that should be scattered in, i.e. one for each of the
        # (scatter_batch_size * num_tokens) word embeddings.
        scatter_indices = []
        for (batch_index, sentence_embs, offset) in zip(batch_indices,
                                                        filtered_embs, offsets):
            for (token_index, _) in enumerate(sentence_embs):
                scatter_indices.append([batch_index, token_index + offset])

        # Scatters passed word embeddings into embeddings gathered from tokens.
        # <tf.float32>[batch_size, num_tokens + num_special_tokens, emb_size]
        return tf.tensor_scatter_nd_update(input_embs, scatter_indices, updates)

    def scatter_all_embeddings(self, inputs, input_embs):
        """Scatters custom passed embeddings for text segment inputs.

        Args:
          inputs: the model inputs, which contain any custom embeddings to scatter.
          input_embs: the default model embeddings.

        Returns:
          The default model embeddings with scattered custom embeddings.
        """
        # Gets batch indices of any word embeddings that were passed for text_a.
        passed_input_embs = [ex.get("input_embs_" + self.config.prompt)
                             for ex in inputs]
        batch_indices = [index for (index, emb) in enumerate(
            passed_input_embs) if emb is not None]

        # If word embeddings were passed in for text_a, scatter them into the
        # embeddings, gathered from the input ids. 1 is passed in as the offset
        # for each, since text_a starts at index 1, after the [CLS] token.
        if batch_indices:
            input_embs = self._scatter_embs(
                passed_input_embs, input_embs, batch_indices,
                offsets=np.ones(len(batch_indices), dtype=np.int64))

        return input_embs

    def get_target_scores(self, inputs: Iterable[JsonDict], scores):
        """Get target-class scores, as a 1D tensor.

        Args:
          inputs: list of input examples
          scores: <tf.float32>[batch_size, num_classes], either logits or probas

        Returns:
          <tf.float32>[batch_size] target scores for each input
        """
        arg_max = tf.math.argmax(scores, axis=-1).numpy()
        # {0: 'H', 1: 'H2', 2: 'HR', 3: 'OK', 4: 'S', 5: 'S3', 6: 'SH', 7: 'V', 8: 'V2'}

        labels_from_model = self.model.config.id2label

        grad_classes = []
        for i, ex in enumerate(inputs):
            label_key = arg_max[i]
            result = next((key for key, value in ex.items() if value == 1), 'OK')
            label = result
            grad_classes.append(label)

        grad_idxs = []
        for label in grad_classes:
            if isinstance(label, str):
                idx = next((key for key, value in labels_from_model.items() if value == label))
            else:
                idx = label
            grad_idxs.append(idx)

        # list of tuples (batch idx, label idx)
        gather_indices = []
        for batch_idx, label_idx in enumerate(grad_idxs):
            gather_indices.append((batch_idx, label_idx))

        # <tf.float32>[batch_size]
        output = tf.gather_nd(scores, gather_indices), grad_idxs
        return output

    ##
    # LIT API implementation
    ##
    def max_minibatch_size(self):
        return self.config.inference_batch_size

    def predict_minibatch(self, inputs: Iterable[JsonDict]):
        # Use watch_accessed_variables to save memory by having the tape do nothing
        # if we don't need gradients.
        with (tf.GradientTape(watch_accessed_variables=self.config.compute_grads) as tape):
            logging.info('-------------------------> using predict here')

            encoded_input = self._preprocess(inputs)  # is a tensor of the encoded input

            # Gathers word embeddings from BERT model embedding layer using input ids
            # of the tokens.
            input_ids = encoded_input["input_ids"]

            word_embeddings = self.model.deberta.embeddings.weight
            # Needed for comparison of weights
            np.savetxt('tf_deberta_weights.txt', word_embeddings)
            logging.info('--> weights are saved')

            # <tf.float32>[batch_size, num_tokens, emb_size]
            # used to slice the input tensor based on the indices provided
            input_embs = tf.gather(word_embeddings, input_ids)

            # Scatter in any passed in embeddings.
            # <tf.float32>[batch_size, num_tokens, emb_size]
            input_embs = self.scatter_all_embeddings(inputs, input_embs)

            tape.watch(input_embs)  # Watch input_embs for gradient calculation.

            model_inputs = encoded_input.copy()
            model_inputs["input_ids"] = None
            out: TFSequenceClassifierOutput = self.model(
                model_inputs,
                inputs_embeds=input_embs,
                training=False,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True)

            batched_outputs = {
                "input_ids": encoded_input["input_ids"],
                "ntok": tf.reduce_sum(encoded_input["attention_mask"], axis=1),
                "cls_emb": out.hidden_states[-1][:, 0],  # last layer, first token
            }

            if self.config.output_embeddings:
                batched_outputs["input_embs"] = input_embs
                self._verify_num_layers(out.hidden_states)  # type is not the same but only numbers in the methods

                # <float32>[batch_size, num_tokens, 1]
                token_mask = tf.expand_dims(
                    tf.cast(encoded_input["attention_mask"], tf.float32), axis=2)

                # <float32>[batch_size, 1]
                denom = tf.reduce_sum(token_mask, axis=1)

                for i, layer_output in enumerate(out.hidden_states):
                    # layer_output is <float32>[batch_size, num_tokens, emb_dim]
                    # average over tokens to get <float32>[batch_size, emb_dim]
                    batched_outputs[f"layer_{i}/avg_emb"] = tf.reduce_sum(
                        layer_output * token_mask, axis=1) / denom

            if self.config.output_attention:
                if len(out.attentions) != self.model.config.num_hidden_layers:
                    raise ValueError("Unexpected size of attentions. Should be the same "
                                     "size as the number of hidden layers. Expected "
                                     f"{self.model.config.num_hidden_layers}, got "
                                     f"{len(out.attentions)}.")

                for i, layer_attention in enumerate(out.attentions):
                    batched_outputs[f"layer_{i + 1}/attention"] = layer_attention

            batched_outputs["probas"] = tf.nn.softmax(out.logits, axis=-1)
            # <tf.float32>[batch_size], a single target per example

            scalar_targets, grad_idxs = self.get_target_scores(
                inputs, batched_outputs["probas"]
            )

            if self.config.compute_grads:
                batched_outputs[self.config.label_names] = tf.convert_to_tensor(
                    grad_idxs
                )

        # Request gradients after the tape is run.
        # Note: embs[0] includes position and segment encodings, as well as subword embeddings.
        if self.config.compute_grads:
            # <tf.float32>[batch_size, num_tokens, emb_dim]
            batched_outputs["input_emb_grad"] = tape.gradient(
                scalar_targets, input_embs
            )

        detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
        # Sequence of dicts, one per example.
        unbatched_outputs = utils.unbatch_preds(detached_outputs)
        return map(self._postprocess, unbatched_outputs)

    def input_spec(self) -> Spec:
        ret = {}
        ret[self.config.prompt] = lit_types.TextSegment()
        ret["tokens_" + self.config.prompt] = lit_types.Tokens(
            parent=self.config.prompt, required=False)

        if self.is_regression:
            ret[self.config.label_names] = lit_types.Scalar(required=False)
        else:
            ret[self.config.label_names] = lit_types.CategoryLabel(
                required=False, vocab=self.config.labels)

        if self.config.output_embeddings:
            # The input_embs_ fields are used for Integrated Gradients.
            text_a_embs = "input_embs_" + self.config.prompt
            ret[text_a_embs] = lit_types.TokenEmbeddings(
                align="tokens", required=False)
        return ret

    def output_spec(self) -> Spec:
        ret = {"tokens": lit_types.Tokens()}
        ret["tokens_" + self.config.prompt] = lit_types.Tokens(parent=self.config.prompt)
        ret["probas"] = lit_types.MulticlassPreds(
            parent=self.config.label_names,
            vocab=self.config.labels,
            null_idx=self.config.null_label_idx)

        if self.config.output_embeddings:
            ret["cls_emb"] = lit_types.Embeddings()
            # Average embeddings, one per layer including embeddings.
            for i in range(1 + self.model.config.num_hidden_layers):
                ret[f"layer_{i}/avg_emb"] = lit_types.Embeddings()

            # The input_embs_ fields are used for Integrated Gradients.
            ret["input_embs_" + self.config.prompt] = lit_types.TokenEmbeddings(
                align="tokens_" + self.config.prompt)

        # Gradients, if requested.
        if self.config.compute_grads:
            ret["cls_grad"] = lit_types.Gradients(
                align=("score" if self.is_regression else "probas"),
                grad_for="cls_emb",
                grad_target_field_key=self.config.label_names,
            )
            if not self.is_regression:
                ret[self.config.label_names] = lit_types.CategoryLabel(
                    required=False, vocab=self.config.labels
                )
            if self.config.output_embeddings:
                text_a_token_grads = "token_grad_" + self.config.prompt
                ret[text_a_token_grads] = lit_types.TokenGradients(
                    align="tokens_" + self.config.prompt,
                    grad_for="input_embs_" + self.config.prompt,
                    grad_target_field_key=self.config.label_names,
                )

        if self.config.output_attention:
            # Attention heads, one field for each layer.
            for i in range(self.model.config.num_hidden_layers):
                ret[f"layer_{i + 1}/attention"] = lit_types.AttentionHeads(
                    align_in="tokens", align_out="tokens")
        return ret


class ModerationModelSpec(ModerationModel):
    def __init__(self, *args, **kw):
        super().__init__(
            *args,
            prompt="prompt",
            label_names='label',
            labels=['H', 'H2', 'HR', 'OK', 'S', 'S3', 'SH', 'V', 'V2'],
            **kw)

    def output_spec(self) -> Spec:
        ret = super().output_spec()
        ret["probas"] = lit_types.MulticlassPreds(
            parent=self.config.label_names,
            vocab=self.config.labels)
        return ret
