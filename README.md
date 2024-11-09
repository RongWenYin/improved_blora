# Improved B-LoRA for Style Preservation and Transformation

![Improved BLora Image](docs/improved_blora.png)

The objective of this repository is to enhance B-LoRA based on insights from InstantStyle, with a focus on layers 2 and 5 to improve style preservation and transformation. This approach uses a single photo as the style reference to ensure consistent style application across various content prompts.


## Getting Started

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
   # Using the specified cust_block_list, only the LoRA layers within attention blocks W2 and W5 will be trained.
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
      --cust_block_list="down_blocks.2.attentions.1 up_blocks.0.attentions.1"

      ```
This will optimize the Improved B-LoRAs weights for the style and store them in  `output_dir`.


2. **Inference**   

   For image stylization based on a reference style image (1), run:
   ```
   from utils import *
   # Use subset of layers for image generation with specific style customizations.
   # Examples include using combining layers such as W5 or W2 and W5.
   layer_list = ['W5','W2_W5']  # Layer combinations for style customization
   object_names = ["girl", "cat", "apple", "dog", "fish"]
   # The inferenceImages function generates images based on a list of object names,
   # leveraging the LoRA model to apply specific styles across all combinations of attention blocks W2 and W5.
   inferenceImages(object_names,layer_list,style_B_LoRA_path,promptKey,styleKey)
   ```
   This will generate new images with the style of the Improved B-LoRAs.

3. **Result** 
![Result1 Image](docs/r1.png)
![Result2 Image](docs/r2.png)
![Result3 Image](docs/r3.png)

## License

This project is licensed under the [MIT License](LICENSE).

