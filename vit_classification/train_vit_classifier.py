import sys
sys.path.append('..')

import torch
from transformers import ViTForImageClassification, ViTConfig
from torchinfo import summary
from convert_celebA import celeba_loader
import configs
import tqdm
from my_utils.accuracy_comp import multi_label_accuracy
from vit_classification.vit_gaussian_pixels import vit_with_embedder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the feature extractor and model from Hugging Face Transformers
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# feature_extractor = ViTFeatureExtractor(image_size=128, do_resize=False, do_center_crop=False, do_normalize=False)
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
# Initialize the ViT configuration tailored for your dataset
config = ViTConfig(
    image_size=configs.celeba_config.img_size,
    patch_size=16,
    num_channels=3,
    hidden_size=768,
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
    model = ViTForImageClassification(config).to(device)
elif configs.celeba_config.model_name.lower() == 'non_uniform_vit':
    setattr(config, 'num_patches', configs.celeba_config.gaussian_pixel.pix_per_img)
    model = vit_with_embedder.ViTNonUniformPatches(config).to(device)
else:
    raise ValueError(f'Unknown model name: {configs.celeba_config.model_name}, possible values: vanilla_vit, non_uniform_vit')

model_summary = summary(model, input_size=(1, 3, configs.celeba_config.img_size, configs.celeba_config.img_size),
                        verbose=1, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=configs.celeba_config.vit_conf.learning_rate)

# Load an image
train_loader, valid_loader, _ = celeba_loader.get_celeba_data_loaders(
    batch_size=configs.celeba_config.vit_conf.batch_size, return_testloader=False)
loss_criterion = torch.nn.BCEWithLogitsLoss()

valid_accuracy = 0.0
moving_avg_weight = 0.3
overfit_batch = False
is_first_batch = True
for epoch in range(configs.celeba_config.vit_conf.epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    pbar = tqdm.tqdm(train_loader)
    for imgs_this_batch, labels_this_batch in pbar:
        if overfit_batch:
            if is_first_batch:
                imgs = imgs_this_batch.to(device)
                labels = labels_this_batch.to(torch.float32).to(device)
                is_first_batch = False
        else:
            imgs = imgs_this_batch.to(device)
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

    with torch.no_grad():
        pbar_valid = tqdm.tqdm(valid_loader, desc="Validation")
        for imgs, labels in pbar_valid:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            logits = outputs.logits

            # Calculate accuracy
            total_predictions_val += labels.size(0)
            correct_predictions_val += (multi_label_accuracy(logits, labels).item() * labels.size(0))

    valid_accuracy = correct_predictions_val / total_predictions_val

    # Print epoch-level summary
    print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {valid_accuracy:.4f}")
