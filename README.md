# Implicit Style-Content Separation using Improved B-LoRAs

![Teaser Image](docs/teaser_blora.png)

This repository contains the official implementation of the Improved B-LoRAs method, which enables implicit style separation of a single input image for various image stylization tasks. Improved B-LoRAs leverages the power of Stable Diffusion XL (SDXL) and Low-Rank Adaptation (LoRA) to disentangle the style components of an image, facilitating applications such as image style transfer, text-based image stylization, and consistent style generation.

## Getting Started

### Prerequisites
- Python 3.11.6+
- PyTorch 2.1.1+
- Other dependencies (specified in `requirements.txt`)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/RongWenYin/improved_blora.git
   cd improved_blora
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. **Training Improved Improved B-LoRAs**

   To train the Improved Improved B-LoRAs for a given input image, run:
   ```
   !accelerate launch train_dreambooth_b-lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --instance_data_dir="./images/{styleKey}" \
    --output_dir="{output_dir}" \
    --instance_prompt="{promptKey}" \
    --resolution=1024 \
    --rank=64 \
    --train_batch_size=1 \
    --learning_rate=5e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --checkpointing_steps=1000 \
    --seed="0" \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision="fp16" \
    --cust_block_list="down_blocks.2.attentions.0 down_blocks.2.attentions.1 mid_block.attentions.0 up_blocks.0.attentions.0 up_blocks.0.attentions.1 up_blocks.0.attentions.2"

      ```
This will optimize the Improved B-LoRAs weights for the style and store them in  `output_dir`.
Parameters that need to replace  `instance_data_dir`, `output_dir`, `instance_prompt` (in our paper we use `A [v]`)


![Apps Image](docs/apps_method1.png)

2. **Inference**   

   For image stylization based on a reference style image (1), run:
   ```
   from utils import *

   style_B_LoRA_path = f'./checkpoint/pytorch_lora_weights.safetensors'
   objectNames = ["girl", "cat", "apple", "dog", "fish"]
   pipeline = None  # Start with pipeline uninitialized


   for layers in layerList:
      genImagesBatch(layers, pipeline, objectNames)
   ```
   This will generate new images with the style of the Improved B-LoRAs.


## License

This project is licensed under the [MIT License](LICENSE).

