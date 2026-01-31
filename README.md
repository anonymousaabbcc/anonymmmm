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

- **Porto**: PKDD 2015 Taxi Trajectory Prediction  
  https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i

- **Beijing**: START repository  
  https://github.com/aptx1231/START

Please follow the original dataset terms and licenses when downloading and using the data.  
Xi’an (XA) and Chengdu (CD) will be released upon paper acceptance.

---

## Backbone Model

We use **Llama 3.2 (1B)** from Hugging Face:  
- Model page: https://huggingface.co/meta-llama/Llama-3.2-1B  
- License text: https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/LICENSE.txt

---

## License Notice

This project uses **Llama 3.2**, which is licensed under the **Llama 3.2 Community License** (see the license text link above).  
Copyright © Meta Platforms, Inc. All Rights Reserved.
