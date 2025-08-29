# Dual-Stream Top View Person Re-Identification
PyTorch code for paper "A Framework for Integrating Person Re-Identification with Retail and Marketing Systems".

- The proposed design in this study is a dual-stream architecture, one for RGB modality and the other for Depth modality, with CNN-based ResNet-50 as the feature extractor  backbone.

- For Generalization Evaluation of the proposed architecture, Transformer-based ViT-Base/16 was also adopted as the feature extractor backbone.

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
- After the data is stored in ".npy" format, you need to define the data path in `config.py` before training or testing.

### 3. Training

- ResNet-50 Backbone
  
  There are two architectures with a ResNet-50 backbone as a feature extractor. One is a dual-stream architecture, and the other is single-stream architecture.

    - To train dual-stream architecture, run the python file `train_dual_stream_net.py` in the Backbone-ResNet folder by executing:
      
    ```bash
    python train_dual_stream_net.py
    ```
    - To train single-stream architecture, run the python file `train_single_stream_net.py` in the Backbone-ResNet folder by executing:
      
    ```bash
    python train_single_stream_net.py
    ```
  
- ViT Backbone

  - To train using ViT backbone as a feature extractor, run the python file `train_ViT.py` in the Backbone-ViT folder by executing:
    
  ```bash
  python train_ViT.py --config_file "config\\config.yml"
  ```

### 4. Testing

- Evaluate a trained model by executing:
  
```bash
  python test.py --resume 'model_path'
  ```
   - `--resume`: the saved model path.

### 5. Requirements

- Python version: `Python==3.9`

- PyTorch version: `torch==1.12.1+cu11.3`

- Other required libraries can be installed by: `pip install requirements.txt`

- All the Hyperparameters can be found in the `config.py` file.

### 6. References

Most of the code of the backbone architecture is borrowed from AGW [1] and TransReID [2].

[1] Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE transactions on pattern analysis and machine intelligence, 2021, 44(6): 2872-2893.

[2] He S, Luo H, Wang P, et al. Transreid: Transformer-based object re-identification[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 15013-15022.
