## Pre-processing & Data protocol

This project follows a **re-identification / metric-learning** setup: the model learns an embedding space where images of the same jaguar identity are close and different identities are far. The input pipeline is built to (1) reduce overfitting with augmentation, (2) evaluate retrieval realistically, and (3) keep validation stable and comparable across folds.

### Data sources and splits

Training metadata comes from `train.csv`, which provides a filename and a jaguar identity label (`ground_truth`). The test set is described by `test.csv`, which contains **query–gallery pairs** `(query_image, gallery_image)` with no labels.

To estimate generalization reliably, I use **Stratified K-Fold cross-validation** on the training set (stratified by identity labels). For `n_folds = K`, each fold uses approximately:

$$

\text{val fraction} \approx \frac{1}{K}, \qquad \text{train fraction} \approx \frac{K-1}{K}.

$$

In code, this is done with `StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)`, so each fold preserves the identity distribution as much as possible. (If `n_folds=2`, validation is 50%; if `n_folds=5`, validation is 20%.)  
An optional `val_size < 1.0` can reduce the validation fold to speed up experiments; effectively:

$$

\text{effective val fraction} \approx \frac{1}{K}\cdot \text{val\_size}.

$$

### Train / Val / Test transform strategy

The pipeline uses **two transform regimes**:

- **Training transforms** are intentionally stochastic and strong to fight overfitting on a limited number of identities/images. They include random resized crops, horizontal flip, color jitter, and random erasing. The idea is to make the embedding robust to viewpoint, illumination, and partial occlusions, while preserving identity cues (spots/rosettes).

- **Evaluation / inference transforms** are deterministic (resize + center crop + normalization). This is critical: retrieval metrics become noisy if we embed validation images under random augmentation.

Concretely, I keep *two views of the training set*:
- `train_loader`: augmented (`train_tf`) for gradient updates
- `train_eval_loader`: non-augmented (`test_tf`) used as the **gallery** when computing embeddings for validation metrics

### Retrieval-style validation protocol (why it matters)

Even though training uses classification-style supervision (ArcFace logits), evaluation should reflect the competition: **similarity ranking**. Therefore validation is done as retrieval:

- **Query set**: embeddings from the validation split  
- **Gallery set**: embeddings from the training split (but with deterministic transforms)

This matches the test-time usage: a query image is compared against a gallery, and ranking quality matters.

### Multi-stage input resolution

Training is run in **stages with increasing input size** (e.g., 384 → 448). The motivation is:
- start cheaper / more stable at the base resolution
- finish at higher resolution to capture fine identity details (rosette patterns)

Each stage rebuilds dataloaders using the stage `img_size`, and keeps deterministic eval transforms consistent with that size. When using backbones that have a preferred resolution (e.g. MegaDescriptor at 384), stages are chosen around that native size.

### Test-time framework

The test set is defined by a list of query–gallery pairs. Inference embeds all unique test images once, then scores each pair using cosine similarity (dot product of L2-normalized embeddings). The submission is a CSV with:

- `row_id` matching `test.csv` order
- `similarity` in \([0, 1]\), typically obtained from cosine similarity \(s \in [-1,1]\) via:

$$

\text{sim}_{01} = \frac{s + 1}{2}.

$$

This keeps preprocessing consistent between validation and test and ensures the model is evaluated under the same retrieval assumptions.

## Model architecture & training objective

The system is a **metric-learning** model trained to produce **L2-normalized embeddings** suitable for retrieval: images of the same jaguar should have high cosine similarity, and different jaguars should be separated in the embedding space.

### Backbone (MegaDescriptor)

The feature extractor is **MegaDescriptor-L** loaded from `timm` via Hugging Face Hub (`BVRA/MegaDescriptor-L-384`). This backbone is designed for instance-level matching and retrieval, and outputs a high-dimensional descriptor (here **1536-d**). Because it is pretrained for descriptor learning, fine-tuning is done conservatively: the backbone is updated with a **smaller learning rate** than the heads (typically 10× smaller) to preserve the pretrained representation while adapting to the jaguar domain.

### Projection / embedding head

A lightweight embedding head is applied on top of backbone features to produce the final descriptor. The head keeps the representation in the same dimension (1536) and ends with **L2-normalization**, so cosine similarity reduces to a dot product:

$$

\hat{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}.

$$

This normalization is essential for stable metric learning and for using cosine similarity consistently at train/val/test time.

*(If the backbone returns spatial maps, the head can include GeM pooling; if it returns a vector descriptor directly, GeM is effectively bypassed.)*

### Classification layer for metric learning (ArcFace)

Although evaluation is retrieval-based, training uses an identity classification surrogate that directly shapes angular margins in embedding space: **ArcFace**. ArcFace replaces a standard linear classifier with a cosine-based classifier and enforces an angular margin \(m\) between classes. With normalized embeddings \(\hat{\mathbf{z}}\) and normalized class weights \(\hat{\mathbf{w}}_c\), the cosine logit for class \(c\) is:

$$

\cos(\theta_c) = \hat{\mathbf{z}}^\top \hat{\mathbf{w}}_c.

$$

For the target class \(y\), ArcFace applies an additive angular margin:

$$

\phi_y = \cos(\theta_y + m),

$$

and scales logits by a constant \(s\) before the softmax. This encourages **tight intra-class clusters** and **larger inter-class separation** directly in cosine space, which aligns well with the competition metric (ranking by similarity).

### Loss function

Training minimizes standard cross-entropy over the ArcFace logits:

$$

\mathcal{L} = -\log \frac{\exp(s \cdot \phi_y)}{\sum_{c}\exp(s \cdot \cos(\theta_c))}.

$$

In practice, the margin \(m\) can be kept fixed (e.g. \(m=0.5\)) for simplicity, or scheduled/ramped across epochs.

### Similarity at inference

At inference, embeddings are L2-normalized and the similarity between query \(q\) and gallery \(g\) is the cosine similarity:

$$

s(q,g) = \hat{\mathbf{z}}_q^\top \hat{\mathbf{z}}_g \in [-1,1].

$$

For the submission format requiring \([0,1]\), the score is mapped as:

$$

\text{sim}_{01} = \frac{s(q,g)+1}{2}.

$$

This keeps training geometry (cosine space) consistent with validation and test-time scoring.
