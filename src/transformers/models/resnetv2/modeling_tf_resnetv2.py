# coding=utf-8
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TensorFlow ResNetv2 model."""

from typing import Dict, Optional, Tuple, Union

import tensorflow as tf

from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithNoAttention,
    TFBaseModelOutputWithPoolingAndNoAttention,
    TFImageClassifierOutputWithNoAttention,
)
from ...modeling_tf_utils import TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable, unpack_inputs
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnetv2 import ResNetv2Config


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetv2Config"
_FEAT_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "nandwalritik/resnetv2"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "nandwalritik/resnetv2"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

TF_RESNETV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nandwalritik/resnetv2",
    # See all ResNetv2 models at https://huggingface.co/models?filter=resnetv2
]


# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetConvLayer with ResNet->ResNetv2
class TFResNetv2ConvLayer(tf.keras.layers.Layer):
    def __init__(
        self, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.pad_value = kernel_size // 2
        self.conv = tf.keras.layers.Conv2D(
            out_channels, kernel_size=kernel_size, strides=stride, padding="valid", use_bias=False, name="convolution"
        )
        # Use same default momentum and epsilon as PyTorch equivalent
        self.normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")
        self.activation = ACT2FN[activation] if activation is not None else tf.keras.layers.Activation("linear")

    def convolution(self, hidden_state: tf.Tensor) -> tf.Tensor:
        # Pad to match that done in the PyTorch Conv2D model
        height_pad = width_pad = (self.pad_value, self.pad_value)
        hidden_state = tf.pad(hidden_state, [(0, 0), height_pad, width_pad, (0, 0)])
        hidden_state = self.conv(hidden_state)
        return hidden_state

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state, training=training)
        hidden_state = self.activation(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetEmbeddings with ResNet->ResNetv2
class TFResNetv2Embeddings(tf.keras.layers.Layer):
    """
    ResNetv2 Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetv2Config, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedder = TFResNetv2ConvLayer(
            config.embedding_size,
            kernel_size=7,
            stride=2,
            activation=config.hidden_act,
            name="embedder",
        )
        self.pooler = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid", name="pooler")
        self.num_channels = config.num_channels

    def call(self, pixel_values: tf.Tensor, training: bool = False) -> tf.Tensor:
        _, _, _, num_channels = shape_list(pixel_values)
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        hidden_state = pixel_values
        hidden_state = self.embedder(hidden_state)
        hidden_state = tf.pad(hidden_state, [[0, 0], [1, 1], [1, 1], [0, 0]])
        hidden_state = self.pooler(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetShortCut with ResNet->ResNetv2
class TFResNetv2ShortCut(tf.keras.layers.Layer):
    """
    ResNetv2 shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, out_channels: int, stride: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.convolution = tf.keras.layers.Conv2D(
            out_channels, kernel_size=1, strides=stride, use_bias=False, name="convolution"
        )
        # Use same default momentum and epsilon as PyTorch equivalent
        self.normalization = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="normalization")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_state = x
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state, training=training)
        return hidden_state


# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetBasicLayer with ResNet->ResNetv2
class TFResNetv2BasicLayer(tf.keras.layers.Layer):
    """
    A classic ResNetv2's residual layer composed by two `3x3` convolutions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.conv1 = TFResNetv2ConvLayer(out_channels, stride=stride, name="layer.0")
        self.conv2 = TFResNetv2ConvLayer(out_channels, activation=None, name="layer.1")
        self.shortcut = (
            TFResNetv2ShortCut(out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else tf.keras.layers.Activation("linear", name="shortcut")
        )
        self.activation = ACT2FN[activation]

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = hidden_state
        hidden_state = self.conv1(hidden_state, training=training)
        hidden_state = self.conv2(hidden_state, training=training)
        residual = self.shortcut(residual, training=training)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetBottleNeckLayer with ResNet->ResNetv2
class TFResNetv2BottleNeckLayer(tf.keras.layers.Layer):
    """
    A classic ResNetv2's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        reduction: int = 4,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.conv0 = TFResNetv2ConvLayer(reduces_channels, kernel_size=1, name="layer.0")
        self.conv1 = TFResNetv2ConvLayer(reduces_channels, stride=stride, name="layer.1")
        self.conv2 = TFResNetv2ConvLayer(out_channels, kernel_size=1, activation=None, name="layer.2")
        self.shortcut = (
            TFResNetv2ShortCut(out_channels, stride=stride, name="shortcut")
            if should_apply_shortcut
            else tf.keras.layers.Activation("linear", name="shortcut")
        )
        self.activation = ACT2FN[activation]

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = hidden_state
        hidden_state = self.conv0(hidden_state, training=training)
        hidden_state = self.conv1(hidden_state, training=training)
        hidden_state = self.conv2(hidden_state, training=training)
        residual = self.shortcut(residual, training=training)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetStage with ResNet->ResNetv2
class TFResNetv2Stage(tf.keras.layers.Layer):
    """
    A ResNetv2 stage composed of stacked layers.
    """

    def __init__(
        self, config: ResNetv2Config, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        layer = TFResNetv2BottleNeckLayer if config.layer_type == "bottleneck" else TFResNetv2BasicLayer

        layers = [layer(in_channels, out_channels, stride=stride, activation=config.hidden_act, name="layers.0")]
        layers += [
            layer(out_channels, out_channels, activation=config.hidden_act, name=f"layers.{i + 1}")
            for i in range(depth - 1)
        ]
        self.stage_layers = layers

    def call(self, hidden_state: tf.Tensor, training: bool = False) -> tf.Tensor:
        for layer in self.stage_layers:
            hidden_state = layer(hidden_state, training=training)
        return hidden_state


# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetEncoder with ResNet->ResNetv2
class TFResNetv2Encoder(tf.keras.layers.Layer):
    def __init__(self, config: ResNetv2Config, **kwargs) -> None:
        super().__init__(**kwargs)
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages = [
            TFResNetv2Stage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
                name="stages.0",
            )
        ]
        for i, (in_channels, out_channels, depth) in enumerate(
            zip(config.hidden_sizes, config.hidden_sizes[1:], config.depths[1:])
        ):
            self.stages.append(TFResNetv2Stage(config, in_channels, out_channels, depth=depth, name=f"stages.{i + 1}"))

    def call(
        self,
        hidden_state: tf.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ) -> TFBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state, training=training)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return TFBaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)


# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetPreTrainedModel with ResNet->ResNetv2,resnet->resnetv2
class TFResNetv2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ResNetv2Config
    base_model_prefix = "resnetv2"
    main_input_name = "pixel_values"

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network. Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        VISION_DUMMY_INPUTS = tf.random.uniform(shape=(3, self.config.num_channels, 224, 224), dtype=tf.float32)
        return {"pixel_values": tf.constant(VISION_DUMMY_INPUTS)}

    @tf.function(
        input_signature=[
            {
                "pixel_values": tf.TensorSpec((None, None, None, None), tf.float32, name="pixel_values"),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)
        return self.serving_output(output)


RESNETV2_START_DOCSTRING = r"""
    This model is a TensorFlow
    [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a
    regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""


RESNETV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@keras_serializable
# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetMainLayer with ResNet->ResNetv2
class TFResNetv2MainLayer(tf.keras.layers.Layer):
    config_class = ResNetv2Config

    def __init__(self, config: ResNetv2Config, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.embedder = TFResNetv2Embeddings(config, name="embedder")
        self.encoder = TFResNetv2Encoder(config, name="encoder")
        self.pooler = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TF 2.0 image layers can't use NCHW format when running on CPU.
        # We transpose to NHWC format and then transpose back after the full forward pass.
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        pixel_values = tf.transpose(pixel_values, perm=[0, 2, 3, 1])
        embedding_output = self.embedder(pixel_values, training=training)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        # Transpose all the outputs to the NCHW format
        # (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
        last_hidden_state = tf.transpose(last_hidden_state, (0, 3, 1, 2))
        pooled_output = tf.transpose(pooled_output, (0, 3, 1, 2))
        hidden_states = ()
        for hidden_state in encoder_outputs[1:]:
            hidden_states = hidden_states + tuple(tf.transpose(h, (0, 3, 1, 2)) for h in hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + hidden_states

        hidden_states = hidden_states if output_hidden_states else None

        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=hidden_states,
        )


@add_start_docstrings(
    "The bare ResNetv2 model outputting raw features without any specific head on top.",
    RESNETV2_START_DOCSTRING,
)
# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetModel with RESNET->RESNETV2,ResNet->ResNetv2,resnet->resnetv2
class TFResNetv2Model(TFResNetv2PreTrainedModel):
    def __init__(self, config: ResNetv2Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.resnetv2 = TFResNetv2MainLayer(config=config, name="resnetv2")

    @add_start_docstrings_to_model_forward(RESNETV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        resnetv2_outputs = self.resnetv2(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return resnetv2_outputs

    def serving_output(
        self, output: TFBaseModelOutputWithPoolingAndNoAttention
    ) -> TFBaseModelOutputWithPoolingAndNoAttention:
        # hidden_states not converted to Tensor with tf.convert_to_tensor as they are all of different dimensions
        return TFBaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=output.hidden_states,
        )


@add_start_docstrings(
    """
    ResNetv2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNETV2_START_DOCSTRING,
)
# Copied from transformers.models.resnet.modeling_tf_resnet.TFResNetForImageClassification with RESNET->RESNETV2,ResNet->ResNetv2,resnet->resnetv2
class TFResNetv2ForImageClassification(TFResNetv2PreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: ResNetv2Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.num_labels = config.num_labels
        self.resnetv2 = TFResNetv2MainLayer(config, name="resnetv2")
        # classification head
        self.classifier_layer = (
            tf.keras.layers.Dense(config.num_labels, name="classifier.1")
            if config.num_labels > 0
            else tf.keras.layers.Activation("linear", name="classifier.1")
        )

    def classifier(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.keras.layers.Flatten()(x)
        logits = self.classifier_layer(x)
        return logits

    @add_start_docstrings_to_model_forward(RESNETV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=TFImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    @unpack_inputs
    def call(
        self,
        pixel_values: tf.Tensor = None,
        labels: tf.Tensor = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        training: bool = False,
    ) -> Union[Tuple[tf.Tensor], TFImageClassifierOutputWithNoAttention]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnetv2(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return TFImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

    def serving_output(self, output: TFImageClassifierOutputWithNoAttention) -> TFImageClassifierOutputWithNoAttention:
        # hidden_states not converted to Tensor with tf.convert_to_tensor as they are all of different dimensions
        return TFImageClassifierOutputWithNoAttention(logits=output.logits, hidden_states=output.hidden_states)
