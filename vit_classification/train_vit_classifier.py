import sys
sys.path.append('..')

import torch
from transformers import ViTForImageClassification, ViTConfig
from torchinfo import summary
from convert_celebA import celeba_loader, gaussian_pix_loader
import configs
import tqdm
from my_utils.accuracy_comp import multi_label_accuracy
from vit_classification.vit_gaussian_pixels import vit_with_embedder
import convert_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the feature extractor and model from Hugging Face Transformers
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# feature_extractor = ViTFeatureExtractor(image_size=128, do_resize=False, do_center_crop=False, do_normalize=False)
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
# Initialize the ViT configuration tailored for your dataset
vit_config = ViTConfig(
    image_size=configs.celeba_config.img_size,
    patch_size=16,
    num_channels=3,
    hidden_size=384,
    num_hidden_layers=12,
    # num_hidden_layers=2,
    num_attention_heads=12,
    # num_attention_heads=2,
    intermediate_size=3072,
    # intermediate_size=512,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    # hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.1,
    # attention_probs_dropout_prob=0.0,
    classifier_dropout=0.1,
    # classifier_dropout=0.0,
    num_labels=configs.celeba_config.classes  # Assuming you are doing classification with 1000 classes
)

# Initialize the model with the custom configuration
if configs.celeba_config.model_name.lower() == 'vanilla_vit':
    model = ViTForImageClassification(vit_config).to(device)
    input_size = (1, 3, configs.celeba_config.img_size, configs.celeba_config.img_size)

elif configs.celeba_config.model_name.lower() == 'non_uniform_vit':
    setattr(vit_config, 'num_patches', configs.celeba_config.gaussian_pixel.pix_per_img)
    setattr(vit_config, 'pixel_dim', 8)
    model = vit_with_embedder.ViTNonUniformPatches(vit_config).to(device)
    input_size = (1, configs.celeba_config.gaussian_pixel.pix_per_img, 8)

else:
    raise ValueError(f'Unknown model name: {configs.celeba_config.model_name}, '
                     f'possible values: vanilla_vit, non_uniform_vit')

if configs.celeba_config.dataloader.lower() == 'gaussian_pix_loader':
    train_loader, valid_loader, _ = gaussian_pix_loader.get_gp_celba_loaders(
        batch_size=configs.celeba_config.vit_conf.batch_size, return_test_loader=False,
        normalize_gaus_params=configs.celeba_config.normalize_gaus_params)
elif configs.celeba_config.dataloader.lower() == 'vanilla_celeba_loader':
    train_loader, valid_loader, _ = celeba_loader.get_celeba_data_loaders(
        batch_size=configs.celeba_config.vit_conf.batch_size, return_testloader=False)
else:
    raise ValueError(f'Unknown dataloader name: {configs.celeba_config.dataloader}, '
                     f'possible values: vanilla_celeba_loader, gaussian_pix_loader')

model_summary = summary(model, input_size=input_size, verbose=1, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=configs.celeba_config.vit_conf.learning_rate)


loss_criterion = torch.nn.BCEWithLogitsLoss()

valid_accuracy = 0.0
moving_avg_weight = 0.3
overfit_batch = False
is_first_batch = True
max_feature = torch.ones((1, 8), device=device) * (-999)  # format mean_x, mean_y, 3 covar, color
min_feature = torch.ones((1, 8), device=device) * 999  # format mean_x, mean_y, 3 covar, color
for epoch in range(configs.celeba_config.vit_conf.epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    pbar = tqdm.tqdm(train_loader)
    for imgs_this_batch, labels_this_batch in pbar:
        imgs_this_batch = imgs_this_batch.to(device)

        if configs.celeba_config.dataloader.lower() == 'gaussian_pix_loader':
            # import ipdb; ipdb.set_trace()
            max_feature = torch.concat([max_feature, imgs_this_batch.view((-1, 8))], dim=0).max(dim=0).values[None, ...]
            min_feature = torch.concat([min_feature, imgs_this_batch.view((-1, 8))], dim=0).min(dim=0).values[None, ...]

        if configs.celeba_config.reconstruct_pixels_from_gaussians:
            x, y, means = convert_images.normalizex_pix_x, convert_images.normalized_px_y, imgs_this_batch[:, :, :2]
            L_params, colors = None, imgs_this_batch[:, :, 5:]
            cov_mat = imgs_this_batch[:, :, [2, 3, 3, 4]].reshape(-1, imgs_this_batch.shape[1], 2, 2)
            imgs_this_batch = convert_images.recon_pix_frm_gaus(x, y, means, L_params, cov_mat, colors)

        if overfit_batch:
            if is_first_batch:
                imgs = imgs_this_batch
                labels = labels_this_batch.to(torch.float32).to(device)
                is_first_batch = False
        else:
            imgs = imgs_this_batch
            labels = labels_this_batch.to(torch.float32).to(device)

        # Forward pass
        outputs = model(imgs)
        logits = outputs.logits
        loss = loss_criterion(logits, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = running_loss * (1 - moving_avg_weight) + moving_avg_weight * loss.item() * labels.size(0)

        # Calculate accuracy
        total_predictions = total_predictions * (1 - moving_avg_weight) + moving_avg_weight * labels.size(0)
        correct_predictions = correct_predictions * (1 - moving_avg_weight) + \
                              moving_avg_weight * multi_label_accuracy(logits, labels).item() * labels.size(0)

        # Calculate running average of accuracy
        train_accuracy = correct_predictions / total_predictions

        # Update progress bar description with the latest training accuracy
        pbar.set_description(f"Epoch: {epoch + 1}, Loss: {running_loss/total_predictions:0.4f}, Batch Train Acc: {train_accuracy:.4f}, "
                             f"Val Acc: {valid_accuracy:.4f}")

    # Validation loop, set model to evaluation mode
    model.eval()
    correct_predictions_val = 0
    total_predictions_val = 0

    print(f'max: {max_feature}, mins: {min_feature}')
    with torch.no_grad():
        pbar_valid = tqdm.tqdm(valid_loader, desc="Validation")
        for imgs, labels in pbar_valid:
            imgs = imgs.to(device)
            labels = labels.to(device)

            if configs.celeba_config.reconstruct_pixels_from_gaussians:
                x, y, means = convert_images.normalizex_pix_x, convert_images.normalized_px_y, imgs[:, :, :2]
                L_params, colors = None, imgs[:, :, 5:]
                cov_mat = imgs[:, :, [2, 3, 3, 4]].reshape(-1, imgs.shape[1], 2, 2)
                imgs = convert_images.recon_pix_frm_gaus(x, y, means, L_params, cov_mat, colors)

            outputs = model(imgs)
            logits = outputs.logits

            # Calculate accuracy
            total_predictions_val += labels.size(0)
            correct_predictions_val += (multi_label_accuracy(logits, labels).item() * labels.size(0))

    valid_accuracy = correct_predictions_val / total_predictions_val

    # Print epoch-level summary
    print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {valid_accuracy:.4f}")
