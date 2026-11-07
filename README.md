# Lost-and-Found
# Lost and Found: AI-Assisted Lost-Item Reporting and Potential-Match Discovery

A Gradio app where people report lost and found items and get a ranked list of possible matches for a human to review. It combines a 12-category image classifier, appearance features, and text similarity. The app suggests candidates and never decides ownership.

Course project at SRM Institute of Science and Technology (Delhi-NCR). First author: Shrestha Chatterjee, who wrote all the code and ran all the experiments. The full write-up is in [`report.pdf`](report.pdf).

## The problem

Searching lost-and-found reports by hand does not scale, and two people describe the same object in different words. An image classifier that predicts only "mug" or "bicycle" cannot show that two reports are the same physical item. This project combines image and text evidence to narrow the search and rank candidates.

## What the app does

| Tab | Purpose |
|---|---|
| Classifier | Shows image-only, description-only and combined predictions for an uploaded image |
| I lost an item | Saves a lost report (image, description, location, contact) with an ID and timestamp |
| I found an item | Saves a found report and flags possible matches against stored lost reports |
| All reports | Shows stored lost and found records side by side |

Reports are saved to a CSV file, with images stored in a Google Drive folder. This is demo-grade storage with no authentication.

## How it works

1. **Classify** the image into one of 12 categories with an ensemble of CNNs trained from scratch.
2. **Describe** the image with appearance features: ResNet34 deep features (768 values), a 64-bin colour histogram and a coarse 12 x 12 colour layout, combined with weights 0.5, 0.3 and 0.2.
3. **Read** the text description with TF-IDF and Logistic Regression, and compute cosine text similarity.
4. **Fuse** the image and text signals with a default weighting of **0.6 image, 0.4 text**.
5. **Rank** candidate reports for a person to review.

## Image classifier

**Data.** [Stanford Online Products](https://www.kaggle.com/datasets/ryanholbrook/stanford-online-products) (Kaggle TFRecords), grouped into 12 super-classes: bicycle, cabinet, chair, coffee maker, fan, kettle, lamp, mug, sofa, stapler, table and toaster.

| Split | Images | Notes |
|---|---|---|
| Train | 53,841 | |
| Validation | 5,710 | Group split on product ID: no product appears in both train and validation |
| Test | 60,502 | Official test split |

**Models.** ResNet18, ResNet34 and RegNet-Y-800MF from torchvision, all trained from random initialisation. EfficientNet-B0 was also trained in the first run.

**Training.**

| Setting | Run 1 | Run 2 (final) |
|---|---|---|
| Input size | 96 x 96 | 112 x 112 |
| Epochs | 30 | 60 |
| Optimiser and loss | AdamW, label-smoothed cross-entropy (0.1) | AdamW, label-smoothed cross-entropy (0.1) |
| Peak learning rate | 2e-3, one-cycle schedule | 2e-3, with schedule |
| Regularisation | Rotation, zoom, shift, flip, brightness and contrast augmentation | Same, plus CutMix (p = 0.6) and EMA of weights (decay 0.998) |
| Evaluation | Flip test-time augmentation, softmax-average ensemble | Flip test-time augmentation, softmax-average ensemble |

## Results

Test accuracy on the 60,502-image test split (12 classes, so chance is about 8%):

| Model | Run 1 | Run 2 (final) |
|---|---|---|
| ResNet18 | 66.34% | 69.24% |
| ResNet34 | 67.09% | 70.27% |
| RegNet-Y-800MF | 65.49% | 69.17% |
| EfficientNet-B0 | 64.38% | not trained |
| **Ensemble** | **69.13%** | **71.93%** |

- Run 2 improved every model by roughly three to four points. The ensemble gained 2.80 points and beat the best single model (ResNet34) by 1.66 points.
- Resolution, training length, CutMix and EMA were changed together, so the gain is not attributed to any one of them.

Per-class results of the final ensemble (test split):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| bicycle | 0.8716 | 0.9271 | 0.8985 |
| cabinet | 0.7103 | 0.7670 | 0.7376 |
| chair | 0.6579 | 0.6531 | 0.6555 |
| coffee maker | 0.7224 | 0.7216 | 0.7220 |
| fan | 0.7249 | 0.6730 | 0.6980 |
| kettle | 0.7770 | 0.7638 | 0.7703 |
| lamp | 0.6777 | 0.7138 | 0.6953 |
| mug | 0.8277 | 0.8628 | 0.8449 |
| sofa | 0.8255 | 0.6774 | 0.7442 |

The remaining classes and the averages are in the notebook's classification report. Bicycle and mug are recognised most reliably. The main confusions are between visually similar furniture classes (cabinet, chair, table, sofa).

## Example outputs

These show behaviour in the app and are not measured performance.

- **Fusion:** for one bicycle image, the image signal gave 46% to bicycle, the description gave 37%, and the combined output gave 67%.
- **Nearest neighbours:** for one query, the six nearest images by appearance were all bicycle parts (similarity 0.81 to 0.88).
- **Report workflow:** a lost pink water bottle was saved with the category "kettle" and no match. When the same bottle was entered as found (kettle, 35%), the app flagged one possible match for review. The wrong category shows that predictions can be imperfect.

## Limitations

- **Matching is not quantitatively evaluated.** No labelled pairs of real lost and found reports exist, so ranking quality is not measured. The 71.93% is category classification accuracy, not matching accuracy.
- **The text model uses synthetic descriptions**, so no text accuracy is claimed.
- **Test-set use during tuning:** Run 1's test accuracy was seen before Run 2 was configured, so the final figure is mildly optimistic.
- Single run per model, no ablations, small 112 x 112 inputs and only 12 broad categories.
- Product photos differ from phone photos of real lost items.

## Future work

- Collect a consent-based set of real lost-and-found photos and descriptions, and measure matching directly (for example top-k recall over known pairs).
- Run ablations for resolution, training length, CutMix and EMA, and repeat with several seeds.
- Compare with pre-trained backbones and with metric-learning or image-text embeddings.
- Add location and time to the ranking, plus authentication and privacy controls.

## Run it

1. Open `lost_and_found.ipynb` in Google Colab with a GPU runtime.
2. Run the data and training cells in order. The dataset downloads through `kagglehub`, and the first load takes a few minutes.
3. The app saves reports to Google Drive (`/content/drive/MyDrive/LostAndFoundUploads`). Change the path if needed.
4. Run the final app cell and open the Gradio link it prints.

## Files

- `lost_and_found.ipynb`: data loading, training, evaluation and the Gradio app
- `report.pdf`: full project report

## Authors

Shrestha Chatterjee (first author) and team. All code and experiments: Shrestha Chatterjee.
