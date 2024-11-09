from diffusers import StableDiffusionXLPipeline, AutoencoderKL 
from huggingface_hub import hf_hub_download
from PIL import Image 
import torch  
import time  
import transformers
from ip_adapter import IPAdapterXL  


# Path to the checkpoint file for the IP adapter model
ip_ckpt = 'ipadapter/sdxl_models/ip-adapter_sdxl.bin'

# Path to the image encoder model used by the IP adapter
image_encoder_path = 'ipadapter/sdxl_models/image_encoder'

# Path to the base model for the Stable Diffusion XL pipeline
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

# The device to run the model on (CUDA is used for GPU acceleration)
device = "cuda"

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

BLOCKS_INST = {
    'W0':['down_blocks.1.attentions.1'],
    'W1':['down_blocks.2.attentions.0'],
    'W2':['down_blocks.2.attentions.1'],
    'W3':['mid_block.attentions.0'],
    'W4':['up_blocks.0.attentions.0'],
    'W5':['up_blocks.0.attentions.1'],
    'W6':['up_blocks.0.attentions.2'],
    'W7':['up_blocks.1.attentions.0'],
    'W8':['up_blocks.1.attentions.1'],
    'W9':['up_blocks.1.attentions.2'],
}



# Function to check if a layer belongs to specific model blocks
def is_belong_to_blocks(key, blocks):
    try:
        # Check if the key (layer name) belongs to any specified blocks
        return any(g in key for g in blocks)
    except Exception as e:
        raise type(e)(f'failed to check if key belongs to blocks, due to: {e}')

# Function to filter LoRA state dictionaries based on specific blocks
def filter_lora(state_dict, blocks_):
    try:
        # Filter state dictionary to include only keys that belong to specified blocks
        return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}
    except Exception as e:
        raise type(e)(f'failed to filter LoRA, due to: {e}')

# Function to scale LoRA state values by a given alpha value
def scale_lora(state_dict, alpha):
    try:
        # Scale each value in the state dictionary by the alpha factor
        return {k: v * alpha for k, v in state_dict.items()}
    except Exception as e:
        raise type(e)(f'failed to scale LoRA, due to: {e}')

# Clear GPU cache by deleting pipeline and clearing memory
def freeCache(pipeline):
    try:
        # Delete the pipeline to free up memory
        del pipeline
    except NameError:
        # If pipeline doesn't exist, do nothing
        print("Pipeline does not exist, no need to delete.")
    except Exception as e:
        print(f"An error occurred when trying to delete pipeline: {e}")
    
    # Clear any remaining GPU memory
    torch.cuda.empty_cache()

# Load and apply style-based LoRA weights to specific model layers
def load_style_to_unet(pipe, layers, style_lora_model_id: str = '', style_alpha: float = 1.1) -> None:
    try:
        # Split the layers to apply and extract corresponding blocks
        layer_list = layers.split('_')
        blocks = []
        for lay in layer_list:
            blocks.extend(BLOCKS_M[lay])

        # Load style-based LoRA weights if specified
        if style_lora_model_id:
            style_B_LoRA_sd, _ = pipe.lora_state_dict(style_lora_model_id)
            print('Applying Improved B-LoRA to blocks:', blocks)
            # Filter and scale LoRA weights according to the blocks and alpha value
            style_B_LoRA = filter_lora(style_B_LoRA_sd, blocks)
            style_B_LoRA = scale_lora(style_B_LoRA, style_alpha)
        else:
            style_B_LoRA = {}
        
        # Load the modified LoRA weights into the model
        pipe.load_lora_into_unet(style_B_LoRA, None, pipe.unet)
    except Exception as e:
        raise type(e)(f'failed to load B-LoRA into unet, due to: {e}')


# Function to generate images for each object in objectNames
def inferenceImages(objectNames,layer_list,style_B_LoRA_path,promptKey,styleKey):
    # Initialize pipeline to None; it will be set during batch generation
    pipeline = None

    # Load the VAE model with specific settings
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)

    # Generate a batch of images for each layer configuration in layer_list
    for layers in layer_list:
        genImagesBatch(layers, pipeline, objectNames, vae,style_B_LoRA_path,promptKey,styleKey)

# Generate images in batch, applying specific styles to each object
def genImagesBatch(layers, pipeline, objectNames, vae, style_B_LoRA_path,promptKey,styleKey,itemstep=1000):

    freeCache(pipeline)  # Clear GPU memory if necessary

    # Initialize the pipeline if it is not yet created
    if pipeline is None:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            vae=vae,
            torch_dtype=torch.float16,
        ).to("cuda")

    # Load the style-based LoRA model into the pipeline
    load_style_to_unet(pipeline, layers, style_B_LoRA_path)
    
    print(f'Generating images for {promptKey} style | {layers} layer | {itemstep} steps')

    # Generate and save images for each object name
    for objectName in objectNames:
        # Define the prompt based on the object and style layer information
        prompt = f'a {objectName} in {promptKey} style | {layers} layer'
        # Run the pipeline to generate the image, with a fixed seed for reproducibility
        image = pipeline(prompt, generator=torch.Generator(device="cuda").manual_seed(138), num_images_per_prompt=1).images[0].resize((512, 512))
        # Save the generated image with a descriptive filename
        image.save(f'{styleKey}__{objectName}__{layers}__{itemstep}.png')
        # Clear GPU memory after saving each image
        torch.cuda.empty_cache()

#Generate images in batch forinstantStyle, applying specific styles to each object
def genImgInst(layerList, styleKey, objectNames):
    refImg = f"./images/{styleKey}/{styleKey}.png"

    for layers in layerList:
        # Split layers into individual block names
        itemLayerList = layers.split('_')
        
        # Collect all the blocks associated with the layers
        blocks = []
        for lay in itemLayerList:
            blocks.extend(BLOCKS_INST[lay])
            
        # Load the SDXL pipeline with reduced memory consumption
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )

        # Enable VAE tiling for memory optimization
        pipe.enable_vae_tiling()
        
        # Load the IP Adapter model
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=blocks)

        # Open and resize the reference image
        image = Image.open(refImg).resize((512, 512))

        # Generate images for each object name
        for objectName in objectNames:
            # Generate image variations using the image prompt
            images = ip_model.generate(
                pil_image=image,
                prompt=f'a {objectName}, masterpiece, best quality, high quality',
                negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                scale=1.0,
                guidance_scale=5,
                num_samples=1,
                num_inference_steps=30,
                seed=42,
            )
            
            # Save the generated image with a descriptive filename
            print(f'Done processing layer: {layers}')
            images[0].save(f'inst__{styleKey}__{objectName}__{layers}__1000.png')
            
            # Free up memory by clearing the pipeline cache
            freeCache(pipe)
