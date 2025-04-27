# Overview of this repo
This repo is an implementation of a vision-language-action model. It takes in image observation, language task command and robot proprioception state, and outputs actions. The model architecture:

```
┌──────────────────────────────┐

│               actions        │

│               ▲              │

│              ┌┴─────┐        │

│  kv cache    │Gemma │        │

│  ┌──────────►│Expert│        │

│  │           │      │        │

│ ┌┴────────┐  │x 10  │        │

│ │         │  └▲──▲──┘        │

│ │PaliGemma│   │  │           │

│ │         │   │  robot state │

│ │         │   noise          │

│ └▲──▲─────┘                  │

│  │  │                        │

│  │  image(s)                 │

│  language tokens             │

└──────────────────────────────┘
```

More specifically, `images` input contain three images. Each image will be tokenized into 256 tokens. The max length of language token is `48`. The robot state contains one token and output action horizon is 50, so there will be 50 action tokens. The order of the token sequence is `[image_token, language_token, state_token, noised_action_token]`. The total sequence length is `256*3 + 48 + 1 + 50 = 867`. If the number of input images is less than 3 or the input language token is more or less than 48, zero pad or truncation will be applied.

# Your task

Your task is to visualize the attention map between the action tokens and the VLM kv cache tokens. The attention weights (logits) has shape: `[num_diffusion_timestep, num_transformer_layer, bsz, num_kv_head, num_query_head, 51, 867]`, where:

- num_diffusion_timestep is set as 10
- num_transformer_layer is set as 18
- bsz is 1 during inference
- num_kv_head is 1 and num_query_head is 8, because the model uses GQA
- 51 is one state token + 50 action tokens
- 867 is the whole sequence length.

You only need to visualize the attention paid by each one of 50 action tokens to the image and language tokens (`256*3+48=816` tokens). The attention weights of each layer and each diffusion timestep should be visualized, and the attention of the same layer&timestep but different attention head should be visualized together. You should use a heat map on images and language tokens for better visualization, as long as there is a head paying attention to a token, the heat should be 'hot'. A reference output file structure:

```
attention_viz/
├── inference_timestep_0
│   ├── diffusion_timestep_0/
│   │   ├── action_token_0/
│   │   │   ├── layer_0/
│   │   │   │   ├── fused_attention_on_base_0_rgb.jpg
│   │   │   │   ├── fused_attention_on_left_wrist_0_rgb.jpg
│   │   │   │   ├── fused_attention_on_right_wrist_0_rgb.jpg
│   │   │   │   ├── fused_attention_on_text.jpg
│   │   │   │   └── ...
│   │   │   ├── layer_1/
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── action_token_1/
│   │   │   └── ...
│   │   └── ...
│   ├── diffusion_timestep_1/
│   │   └── ...
│   └── ...
├── inference_timestep_1/
... 
```

One more thing: the input of the model may only contains 2 images. So in such case, you only need to visualize the attention map of the two image, since the last 256 of the image tokens are padded. The image are tokenized in a rasterization order - from left to right and from up to down.

This is a big and complicated project. So do things carefully as a professional, rigorous and careful software engineer. You may well write the whole visualization pipeline in a separate script and import the visualization function to examples/libero/main.py and takes the logits, and input image+language and outputs the attention map visualization. Most of the reference code you need is in gemma.py and pi0.py.
