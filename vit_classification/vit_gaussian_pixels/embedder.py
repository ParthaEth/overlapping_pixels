import torch
from torch import nn
from transformers import ViTConfig
# from transformers.models.vit.modeling_vit import ViTPatchEmbeddings
#
# import math


class ViTEmbeddingsGaussianPixels(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        # self.patch_embeddings = nn.Identity()  # We simply take the patch descriptors sent to the model
        # self.patch_embeddings = ViTPatchEmbeddings(config)  # todo: Remove this line
        num_patches = config.num_patches
        if num_patches > 256*256:
            raise ValueError(f'num_patches should be less than or equal to 256*256, else define a larger pos '
                             f'embeding param below')
        self.position_embeddings = nn.Parameter(torch.randn(1, config.hidden_size, 256, 256))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos=None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, n_pix, descriptor_size = pixel_values.shape
        # embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding)  #todo :remove this line
        # shape: b, num_patches, hidden_size. num_patches = imgsize/patchsize * imgsize/patchsize
        # hidden_size = patch descriptor length (16X16, e.g. 768 for ViT-B/16)
        embeddings = pixel_values  # We simply take the patch descriptors sent to the model  #todo: add this line

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # In case of Gaussian Pixel, the features of each pixel is [mean (2), L_params(3), color(3)]
        # Therefore we must gridsample the embeddings to get the correct position embeddings
        positional_embedding = torch.nn.functional.grid_sample(
            self.position_embeddings.expand(batch_size, -1, -1, -1),
            embeddings[:, :, None, :2],
            mode='bilinear', align_corners=True)

        # import ipdb; ipdb.set_trace()
        positional_embedding = positional_embedding.squeeze(-1).permute(0, 2, 1)

        # add positional encoding to each token
        embeddings = embeddings + positional_embedding

        embeddings = self.dropout(embeddings)

        return embeddings