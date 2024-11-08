# Import necessary libraries
from diffusers import StableDiffusionXLPipeline, AutoencoderKL  # For using Stable Diffusion with LoRA integration
from huggingface_hub import hf_hub_download
from PIL import Image  # For handling image processing
import torch  # For tensor operations and GPU management
import time  # For tracking time during generation
import transformers  # For model handling in Transformers

# Initialize customizable parameters
styleKey = 'grape'  # Identifier for the style theme (e.g., LoRA model type)
promptKey = "[s90]"  # Key to indicate specific prompt style or category
output_dir = f'outputs_{styleKey}'  # Output directory for saving generated images
style_B_LoRA_path = f'{output_dir}/pytorch_lora_weights.safetensors'  # Path to B-LoRA model weights

# Define objects to generate images for
objectNames = ["girl", "cat", "apple", "dog", "fish"]  # Objects for which images will be generated

# Initialize pipeline and layer list for model tuning
layerList = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W1_W2', 'W1_W3', 'W1_W4', 'W1_W5', 'W1_W6', 'W2_W3', 'W2_W4', 'W2_W5', 'W2_W6', 'W3_W4', 'W3_W5', 'W3_W6', 'W4_W5', 'W4_W6', 'W5_W6']  # Combinations of layers for style customization

# Define blocks to target specific model components for fine-tuning
BLOCKS_M = {
    'content': ['unet.up_blocks.0.attentions.0'],
    'style': ['unet.up_blocks.0.attentions.1', 'unet.up_blocks.0.attentions.2'],
    'W1': ['unet.down_blocks.2.attentions.0'],
    'W2': ['unet.down_blocks.2.attentions.1'],
    'W3': ['unet.mid_block.attentions.0'],
    'W4': ['unet.up_blocks.0.attentions.0'],
    'W5': ['unet.up_blocks.0.attentions.1'],
    'W6': ['unet.up_blocks.0.attentions.2'],
    # Additional layer and block mappings for fine-tuning customization
}


# Function to check if a layer belongs to specific model blocks
def is_belong_to_blocks(key, blocks):
    try:
        return any(g in key for g in blocks)
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')

# Function to filter LoRA state dictionaries based on specific blocks
def filter_lora(state_dict, blocks_):
    try:
        return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}
    except Exception as e:
        raise type(e)(f'failed to filter_lora, due to: {e}')

# Function to scale LoRA state values by a given alpha value
def scale_lora(state_dict, alpha):
    try:
        return {k: v * alpha for k, v in state_dict.items()}
    except Exception as e:
        raise type(e)(f'failed to scale_lora, due to: {e}')

# Clear GPU cache by deleting pipeline and clearing memory
def freeCache(pipeline):
    try:
        del pipeline  # Remove pipeline object to free memory
    except NameError:
        print("Pipeline does not exist, no need to delete.")
    except Exception as e:
        print(f"An error occurred when trying to delete pipeline: {e}")
    
    torch.cuda.empty_cache()  # Clear any remaining GPU memory

# Load and apply style-based LoRA weights to specific model layers
def load_style_to_unet(pipe, layers, style_lora_model_id: str = '', style_alpha: float = 1.1) -> None:
    try:
        layerList = layers.split('_')
        blocks = []
        for lay in layerList:
            blocks.extend(BLOCKS_M[lay])

        if style_lora_model_id:
            style_B_LoRA_sd, _ = pipe.lora_state_dict(style_lora_model_id)
            print('Applying Improved B-LoRA to blocks:', blocks)
            style_B_LoRA = filter_lora(style_B_LoRA_sd, blocks)
            style_B_LoRA = scale_lora(style_B_LoRA, style_alpha)
        else:
            style_B_LoRA = {}
        
        pipe.load_lora_into_unet(style_B_LoRA, None, pipe.unet)
    except Exception as e:
        raise type(e)(f'failed to load_b_lora_to_unet, due to: {e}')


def inferenceImages(objectNames):
    # Initialize the pipeline as None; it will be initialized during batch generation
    pipeline = None

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)

    # Loop through each layer configuration in layerList and generate a batch of images for each
    for layers in layerList:
        genImagesBatch(layers, pipeline, objectNames,vae)

# Generate images in batch, applying specific styles to each object
def genImagesBatch(layers, pipeline, objectNames, vae, itemstep=1000):

    freeCache(pipeline)  # Clear memory if necessary

    # Initialize the pipeline if not yet created
    if pipeline is None:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
        ).to("cuda")

    # Load the style into the model
    load_style_to_unet(pipeline, layers, style_B_LoRA_path)
    
    print(f'Generating images for {promptKey} style | {layers} layer | {itemstep} steps')

    # Generate and save images
    for objectName in objectNames:
        prompt = f'a {objectName} in {promptKey} style | {layers} layer'
        image = pipeline(prompt, generator=torch.Generator(device="cuda").manual_seed(138), num_images_per_prompt=1).images[0].resize((512, 512))
        image.save(f'{styleKey}__{objectName}__{layers}__{itemstep}.png')
        torch.cuda.empty_cache()  # Free GPU memory after saving each image
