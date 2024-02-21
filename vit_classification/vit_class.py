import sys
sys.path.append('..')

import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTConfig
from torchinfo import summary
from convert_celebA import celeba_loader
import configs
import tqdm
from my_utils.accuracy_comp import multi_label_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the feature extractor and model from Hugging Face Transformers
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
feature_extractor = ViTFeatureExtractor(image_size=128, do_resize=False, do_center_crop=False, do_normalize=False)
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
# Initialize the ViT configuration tailored for your dataset
config = ViTConfig(
    image_size=configs.celeba_config.img_size,
    patch_size=16,
    num_channels=3,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    classifier_dropout=0.1,
    num_labels=configs.celeba_config.classes  # Assuming you are doing classification with 1000 classes
)

# Initialize the model with the custom configuration
model = ViTForImageClassification(config).to(device)
model_summary = summary(model, input_size=(1, 3, configs.celeba_config.img_size, configs.celeba_config.img_size),
                        verbose=1, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=configs.celeba_config.vanilla_vit.learning_rate)

# Load an image
train_loader, valid_loader, test_loader = celeba_loader.get_celeba_data_loaders(batch_size=configs.batch_size)
loss_criterion = torch.nn.BCEWithLogitsLoss()

valid_accuracy = 0.0
moving_avg_weight = 0.3
for epoch in range(configs.celeba_config.vanilla_vit.epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    pbar = tqdm.tqdm(train_loader)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(torch.float32).to(device)

        # Forward pass
        outputs = model(imgs)
        logits = outputs.logits
        loss = loss_criterion(logits, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        total_predictions = total_predictions * (1 - moving_avg_weight) + moving_avg_weight * labels.size(0)
        correct_predictions = correct_predictions * (1 - moving_avg_weight) + \
                              moving_avg_weight * multi_label_accuracy(logits, labels).item()

        # Calculate running average of accuracy
        train_accuracy = correct_predictions / total_predictions

        # Update progress bar description with the latest training accuracy
        pbar.set_description(f"Epoch: {epoch + 1}, Batch Train Acc: {train_accuracy:.4f}, "
                             f"Val Acc: {valid_accuracy:.4f}")

    # Validation loop, set model to evaluation mode
    model.eval()
    correct_predictions_val = 0
    total_predictions_val = 0

    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            logits = outputs.logits

            # Calculate accuracy
            total_predictions_val += labels.size(0)
            correct_predictions_val += multi_label_accuracy(logits, labels).item()

    valid_accuracy = correct_predictions_val / total_predictions_val

    # Print epoch-level summary
    print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {valid_accuracy:.4f}")
