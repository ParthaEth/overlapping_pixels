import torch
import torch.nn as nn
from transformers.modeling_outputs import ImageClassifierOutput
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import ViTConfig, ViTPreTrainedModel
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTEncoder, ViTPooler, ViTPatchEmbeddings
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Dict, List
from .embedder import ViTEmbeddingsGaussianPixels


class ViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddingsGaussianPixels(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        print('We use identity as input embeddings, Gaussian pixels are of size b, N, 8, '
              '(mean(2), l_params(3), color(3))')
        return

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values = None,
        bool_masked_pos = None,
        head_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        interpolate_pos_encoding = None,
        return_dict = None,
    ):
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if pixel_values.dim() != 3:
            # print('Pixel values are of shape b, N, 8, (mean(2), l_params(3), color(3))')
            batch_size, channels, height, width = pixel_values.shape
            # Calculate the number of patches along height and width dimensions
            n_patches_side = height // self.config.patch_size

            # Extract patches using unfold
            patches = pixel_values.unfold(2, self.config.patch_size, self.config.patch_size).unfold(3, self.config.patch_size, self.config.patch_size)
            patches = patches.contiguous().view(batch_size, channels, n_patches_side * n_patches_side,
                                                self.config.patch_size * self.config.patch_size)
            patches = patches.permute(0, 2, 1, 3).contiguous().view(batch_size, n_patches_side * n_patches_side, -1)

            # Calculate the number of patches along each dimension
            n_patches_x = height // self.config.patch_size
            n_patches_y = width // self.config.patch_size

            # Generate evenly spaced coordinates for each patch
            x = torch.linspace(0, 1, n_patches_x)
            y = torch.linspace(0, 1, n_patches_y)

            # Create a meshgrid of x, y coordinates
            xx, yy = torch.meshgrid(x, y, indexing='ij')

            # Flatten the meshgrid coordinates
            x_flat = xx.flatten()
            y_flat = yy.flatten()

            # Combine the flattened x and y coordinates into a single tensor
            # Each row in coords represents the (x, y) coordinates of a patch
            coords = torch.stack((x_flat, y_flat), dim=1).unsqueeze(0).expand(batch_size, -1, -1)
            patches[:, :, :2] = coords

            pixel_values = patches

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTNonUniformPatches(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size,
                                    config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
            self,
            pixel_values = None,
            head_mask = None,
            labels = None,
            output_attentions = None,
            output_hidden_states = None,
            interpolate_pos_encoding = None,
            return_dict = None,):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
