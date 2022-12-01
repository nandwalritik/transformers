# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

# rely on isort to merge the imports
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available


_import_structure = {
    "configuration_resnetv2": ["RESNETV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "ResNetv2Config", "ResNetv2OnnxConfig"]
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_resnetv2"] = [
        "RESNETV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ResNetv2ForImageClassification",
        "ResNetv2Model",
        "ResNetv2PreTrainedModel",
        "ResNetv2Backbone",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_resnetv2"] = [
        "TF_RESNETV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFResNetv2ForImageClassification",
        "TFResNetv2Model",
        "TFResNetv2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_resnetv2 import RESNETV2_PRETRAINED_CONFIG_ARCHIVE_MAP, ResNetv2Config, ResNetv2OnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_resnetv2 import (
            RESNETV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            ResNetv2Backbone,
            ResNetv2ForImageClassification,
            ResNetv2Model,
            ResNetv2PreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_resnetv2 import (
            TF_RESNETV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFResNetv2ForImageClassification,
            TFResNetv2Model,
            TFResNetv2PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
