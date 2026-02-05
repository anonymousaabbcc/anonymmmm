# README

This repository contains the codebase and dataset links for our paper.

## Project Structure

- `project/`
  - `CL_embed_combined/`  
    **Trajectory Contrastive Learning** component. Produces trajectory embeddings used by later stages.
  - `Graph_based_prompt_align/`  
    **Trajectory–Text Representation Alignment** component. Aligns trajectory embeddings with text/prompt representations.
  - `Multi_cities_prediction/`  
    **Multi-City Knowledge Integration** component. Performs multi-city training and prediction using the aligned representations.

---

## Data Sources

We use the following public datasets:

- **Porto**:
  https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i

- **Beijing**: 
  https://github.com/aptx1231/START

- **Chengdu (CD)**:  
  https://github.com/Whale2021/Dataset

- **Xi’an (XA)** (either of the following sources):  
  https://www.kaggle.com/datasets/ash1971/didi-dataset  
  https://github.com/mzy94/JCLRNT

Please follow the original dataset terms and licenses when downloading and using the data.

---

## Data Preprocessing

- Store each trajectory as a fixed-length sequence of `(x, y, t, mask)`, where `mask=1` indicates a real point and `mask=0` indicates padding.
- Discard trajectories with fewer than 8 observed points.
- Pad all trajectories to a unified length `T_max = max(max_len(train), max_len(test))`; padded steps are zeroed out and should be ignored during evaluation.
- Normalize `(x, y, t)` to `[-1, 1]` per city using min–max fitted on the training split, and apply the same scaler to test.

---

## Backbone Model

We use **Llama 3.2 (1B)** from Hugging Face:  
- Model page: https://huggingface.co/meta-llama/Llama-3.2-1B  
- License text: https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/LICENSE.txt

---

## License Notice

This project uses **Llama 3.2**, which is licensed under the **Llama 3.2 Community License** (see the license text link above).  
Copyright © Meta Platforms, Inc. All Rights Reserved.
