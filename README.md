# Person-Re-Identification-in-Retail-and-Marketing-Systems
PyTorch code for paper "A Framework for Integrating Person Re-Identification with Retail and Marketing Systems".

I adopt the CNN-based ResNet-50 and Transformer-based ViT-Base/16 as backbones, respectively.

### 1. Results

|Backbone    | Pretrained| Rank@1   | mAP      | Model|
| --------   | -----     | -----    |--------- |------|
|ResNet-50   | ImageNet  | ~ 97.94% | ~ 98.82% | [GoogleDrive](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)|
|ViT         | ImageNet  | ~ 83.20% | ~ 91.20% | [GoogleDrive](https://drive.google.com/open?id=181K9PQGnej0K5xNX9DRBDPAf3K9JosYk)|

### 2. Datasets

- The dataset in the form of images and in .npy format can be downloaded from this link.
- The link contains training data and testing data as well for evaluation purposes.

#### Prepare Custom Dataset

- If you want to prepare your own custom dataset, you need to run `python process_dataset.py` to prepare the dataset, the training data will be stored in ".npy" format.
- After the data is stored in ".npy" format, you need to define the data path in `config.py`.

### 3. Training

- ResNet-50 Backbone
There are two architectures with a ResNet-50 backbone as a feature extractor. One is a dual-stream architecture, and the other is single-stream architecture.

  - To train dual-stream architecture, run the python file `train_dual_stream_net.py` in the Backbone-ResNet folder by executing:
  ```bash
  python train_dual_stream_net.py

  - To train single-stream architecture, run the python file `train_single_stream_net.py` in the Backbone-ResNet folder by executing:
  ```bash
  python train_single_stream_net.py
