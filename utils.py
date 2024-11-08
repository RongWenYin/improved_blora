from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import torch
import time
import transformers

#initial variables, should be changed in jupyterNotebook
styleKey='grape'
promptKey = "[s90]"
output_dir = f'outputs_{styleKey}'

style_B_LoRA_path = f'{output_dir}/checkpoint-1000/pytorch_lora_weights.safetensors'

objectNames = ["girl", "cat", "apple", "dog", "fish"]
pipeline = None
layerList = ['W1','W2','W3','W4','W5','W6','W1_W2','W1_W3','W1_W4','W1_W5','W1_W6','W2_W3','W2_W4','W2_W5','W2_W6','W3_W4','W3_W5','W3_W6','W4_W5','W4_W6','W5_W6']

BLOCKS_M = {
    'content': ['unet.up_blocks.0.attentions.0'],
    'style': ['unet.up_blocks.0.attentions.1','unet.up_blocks.0.attentions.2'],
    'W1':['unet.down_blocks.2.attentions.0'],
    'W2':['unet.down_blocks.2.attentions.1'],
    'W3':['unet.mid_block.attentions.0'],
    'W4':['unet.up_blocks.0.attentions.0'],
    'W5':['unet.up_blocks.0.attentions.1'],
    'W6':['unet.up_blocks.0.attentions.2'],
    'W20':['down_blocks.2.attentions.1.transformer_blocks.0'],
    'W21':['down_blocks.2.attentions.1.transformer_blocks.1'],
    'W22':['down_blocks.2.attentions.1.transformer_blocks.2'],
    'W23':['down_blocks.2.attentions.1.transformer_blocks.3'],
    'W24':['down_blocks.2.attentions.1.transformer_blocks.4'],
    'W25':['down_blocks.2.attentions.1.transformer_blocks.5'],
    'W26':['down_blocks.2.attentions.1.transformer_blocks.6'],
    'W27':['down_blocks.2.attentions.1.transformer_blocks.7'],
    'W28':['down_blocks.2.attentions.1.transformer_blocks.8'],
    'W29':['down_blocks.2.attentions.1.transformer_blocks.9'],
    'W50':['unet.up_blocks.0.attentions.1.transformer_blocks.0'],
    'W51':['unet.up_blocks.0.attentions.1.transformer_blocks.1'],
    'W52':['unet.up_blocks.0.attentions.1.transformer_blocks.2'],
    'W53':['unet.up_blocks.0.attentions.1.transformer_blocks.3'],
    'W54':['unet.up_blocks.0.attentions.1.transformer_blocks.4'],
    'W55':['unet.up_blocks.0.attentions.1.transformer_blocks.5'],
    'W56':['unet.up_blocks.0.attentions.1.transformer_blocks.6'],
    'W57':['unet.up_blocks.0.attentions.1.transformer_blocks.7'],
    'W58':['unet.up_blocks.0.attentions.1.transformer_blocks.8'],
    'W59':['unet.up_blocks.0.attentions.1.transformer_blocks.9'],
#     'style': ['down_blocks.2.attentions.1','up_blocks.0.attentions.0','unet.up_blocks.0.attentions.1'],
}



def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
#                 print('add key..',key)
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')


def filter_lora(state_dict, blocks_):
    try:
        return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}
    except Exception as e:
        raise type(e)(f'failed to filter_lora, due to: {e}')


def scale_lora(state_dict, alpha):
    try:
        return {k: v * alpha for k, v in state_dict.items()}
    except Exception as e:
        raise type(e)(f'failed to scale_lora, due to: {e}')


def get_target_modules(unet, blocks=None):
    try:
        if not blocks:
            blocks = [('.').join(blk.split('.')[1:]) for blk in BLOCKS['content'] + BLOCKS['style']]

        attns = [attn_processor_name.rsplit('.', 1)[0] for attn_processor_name, _ in unet.attn_processors.items() if
                 is_belong_to_blocks(attn_processor_name, blocks)]

        target_modules = [f'{attn}.{mat}' for mat in ["to_k", "to_q", "to_v", "to_out.0"] for attn in attns]
        return target_modules
    except Exception as e:
        raise type(e)(f'failed to get_target_modules, due to: {e}')


# Function to free up GPU memory
def freeCache(pipeline):
    try:
        del pipeline  # Delete the pipeline object if it exists
        # print("Pipeline deleted to free up GPU memory.")
    except NameError:
        print("Pipeline does not exist, no need to delete.")
    except Exception as e:
        print(f"An error occurred when trying to delete pipeline: {e}")
    
    torch.cuda.empty_cache()  # Clear the GPU memory
    
        
def load_style_to_unet(pipe,layers, style_lora_model_id: str = '', style_alpha: float = 1.1) -> None:
        try:
            layerList=layers.split('_')
            blocks=[]
            for lay in layerList:
                blocks.extend(BLOCKS_M[lay])
            # Get Style B-LoRA SD
            if style_lora_model_id:
                style_B_LoRA_sd, _ = pipe.lora_state_dict(style_lora_model_id)
                print('use b-lorra..',blocks)
                style_B_LoRA = filter_lora(style_B_LoRA_sd, blocks)
                style_B_LoRA = scale_lora(style_B_LoRA, style_alpha)
            else:
                style_B_LoRA = {}
            
            pipe.load_lora_into_unet(style_B_LoRA, None, pipe.unet)
        except Exception as e:
            raise type(e)(f'failed to load_b_lora_to_unet, due to: {e}')



def genImagesBatch(layers, pipeline, objectNames,itemstep=1000):    
    # Clear memory if necessary
    freeCache(pipeline)

    # Initialize the pipeline
    if pipeline is None:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
        ).to("cuda")

    # Load style into the model
    load_style_to_unet(pipeline, layers, style_B_LoRA_path)
    
    print(f' {promptKey} style | {layers} layer | {itemstep} step')

    # Generate images
    for objectName in objectNames:
        prompt = f'a {objectName} in {promptKey} style | {layers} layer '
        # print(prompt)
        image = pipeline(prompt, generator=torch.Generator(device="cuda").manual_seed(138), num_images_per_prompt=1).images[0].resize((512, 512))
        image.save(f'{styleKey}__{objectName}__{layers}__{itemstep}.png')
        torch.cuda.empty_cache()  # Free up GPU memory after saving each image



# layerList=['W2_W5']
# layerList=['W5']
# layerList = ['W1','W2','W3','W4','W5','W6']
# layerList = ['W2','W5','W1_W2','W1_W5','W2_W5','W3_W5','W4_W5','W5_W6']
# layerList = ['W1_W2','W1_W3','W1_W4','W1_W5','W1_W6','W2_W3','W2_W4','W2_W5','W2_W6','W3_W4','W3_W5','W3_W6','W4_W5','W4_W6','W5_W6']
# layerList = ['W1','W3','W4','W6','W1_W3','W1_W4','W1_W6','W2_W3','W2_W4','W2_W6','W3_W4','W3_W6','W4_W6']
#
# layerList = ['W1','W2','W3','W4','W5','W6','W1_W2','W1_W3','W1_W4','W1_W5','W1_W6','W2_W6','W3_W6','W4_W6','W5_W6']

# layerList = ['W1_W2_W3','W1_W2_W4','W1_W2_W5','W1_W2_W6','W1_W3_W4','W1_W3_W5','W1_W3_W6','W1_W4_W5','W1_W4_W6','W1_W5_W6','W2_W3_W4','W2_W3_W5','W2_W3_W6','W2_W4_W5','W2_W4_W6','W2_W5_W6','W3_W4_W5','W3_W4_W6','W3_W5_W6','W4_W5_W6']
# layerList = ['W1_W2','W1_W3','W1_W4','W1_W5','W1_W6','W2_W3','W2_W4','W2_W5','W2_W6','W3_W4','W3_W5','W3_W6','W4_W5','W4_W6','W5_W6','W1_W2_W3','W1_W2_W4','W1_W2_W5','W1_W2_W6','W1_W3_W4','W1_W3_W5','W1_W3_W6','W1_W4_W5','W1_W4_W6','W1_W5_W6','W2_W3_W4','W2_W3_W5','W2_W3_W6','W2_W4_W5','W2_W4_W6','W2_W5_W6','W3_W4_W5','W3_W4_W6','W3_W5_W6','W4_W5_W6']
# layerList = ['W1','W2','W3','W4','W5','W6','W1_W2','W1_W3','W1_W4','W1_W5','W1_W6','W2_W3','W2_W4','W2_W5','W2_W6','W3_W4','W3_W5','W3_W6','W4_W5','W4_W6','W5_W6','W1_W2_W3','W1_W2_W4','W1_W2_W5','W1_W2_W6','W1_W3_W4','W1_W3_W5','W1_W3_W6','W1_W4_W5','W1_W4_W6','W1_W5_W6','W2_W3_W4','W2_W3_W5','W2_W3_W6','W2_W4_W5','W2_W4_W6','W2_W5_W6','W3_W4_W5','W3_W4_W6','W3_W5_W6','W4_W5_W6']

#W2345:
# layerList = ['W2','W3','W4','W5','W2_W3','W2_W4','W2_W5','W3_W4','W3_W5','W4_W5','W2_W3_W4','W2_W3_W5','W2_W4_W5','W3_W4_W5','W2_W3_W4_W5']
# layerList = ['W2_W3','W2_W4','W2_W5','W3_W4','W3_W5','W4_W5','W2_W3_W4','W2_W3_W5','W2_W4_W5','W3_W4_W5','W2_W3_W4_W5']


# objectNames = ["girl", "vase", "table", "tower", "fish"]
# layerList = ['W2_W5','W50','W51','W52','W53','W54','W55','W56','W57','W58','W59']
# layerList = ['W2','W5','W50_W51_W52_W53_W54_W55','W55_W56_W57_W58_W59']
# layerList = ['W50_W51_W52_W53_W54_W55_W56_W57_W58_W59','W51_W50_W52_W53_W54_W55_W56_W57_W58_W59','W55_W56_W57_W58_W59_W51_W50_W52_W53_W54','W51_W53_W57_W59_W55']

# layerList=[
# 'W50','W50_W51','W50_W51_W52','W50_W51_W52_W53','W50_W51_W52_W53_W54','W50_W51_W52_W53_W54_W55','W50_W51_W52_W53_W54_W55_W56','W50_W51_W52_W53_W54_W55_W56_W57','W50_W51_W52_W53_W54_W55_W56_W57_W58','W50_W51_W52_W53_W54_W55_W56_W57_W58_W59',
# 'W51','W51_W52','W51_W52_W53','W51_W52_W53_W54','W51_W52_W53_W54_W55','W51_W52_W53_W54_W55_W56','W51_W52_W53_W54_W55_W56_W57','W51_W52_W53_W54_W55_W56_W57_W58','W51_W52_W53_W54_W55_W56_W57_W58_W59',
# 'W52','W52_W53','W52_W53_W54','W52_W53_W54_W55','W52_W53_W54_W55_W56','W52_W53_W54_W55_W56_W57','W52_W53_W54_W55_W56_W57_W58','W52_W53_W54_W55_W56_W57_W58_W59',
# 'W53','W53_W54','W53_W54_W55','W53_W54_W55_W56','W53_W54_W55_W56_W57','W53_W54_W55_W56_W57_W58','W53_W54_W55_W56_W57_W58_W59',
# 'W54','W54_W55','W54_W55_W56','W54_W55_W56_W57','W54_W55_W56_W57_W58','W54_W55_W56_W57_W58_W59',
# 'W55','W55_W56','W55_W56_W57','W55_W56_W57_W58','W55_W56_W57_W58_W59',
# 'W56','W56_W57','W56_W57_W58','W56_W57_W58_W59',
# 'W57','W57_W58','W57_W58_W59',
# 'W58','W58_W59'
# ]

# layerList=[
# 'W20','W20_W21','W20_W21_W22','W20_W21_W22_W23','W20_W21_W22_W23_W24','W20_W21_W22_W23_W24_W25','W20_W21_W22_W23_W24_W25_W26','W20_W21_W22_W23_W24_W25_W26_W27','W20_W21_W22_W23_W24_W25_W26_W27_W28','W20_W21_W22_W23_W24_W25_W26_W27_W28_W29',
# 'W21','W21_W22','W21_W22_W23','W21_W22_W23_W24','W21_W22_W23_W24_W25','W21_W22_W23_W24_W25_W26','W21_W22_W23_W24_W25_W26_W27','W21_W22_W23_W24_W25_W26_W27_W28','W21_W22_W23_W24_W25_W26_W27_W28_W29',
# 'W22','W22_W23','W22_W23_W24','W22_W23_W24_W25','W22_W23_W24_W25_W26','W22_W23_W24_W25_W26_W27','W22_W23_W24_W25_W26_W27_W28','W22_W23_W24_W25_W26_W27_W28_W29',
# 'W23','W23_W24','W23_W24_W25','W23_W24_W25_W26','W23_W24_W25_W26_W27','W23_W24_W25_W26_W27_W28','W23_W24_W25_W26_W27_W28_W29',
# 'W24','W24_W25','W24_W25_W26','W24_W25_W26_W27','W24_W25_W26_W27_W28','W24_W25_W26_W27_W28_W29',
# 'W25','W25_W26','W25_W26_W27','W25_W26_W27_W28','W25_W26_W27_W28_W29',
# 'W26','W26_W27','W26_W27_W28','W26_W27_W28_W29',
# 'W27','W27_W28','W27_W28_W29',
# 'W28','W28_W29'
# ]
# layerList=[
# 'W2_W50','W2_W50_W51','W2_W50_W51_W52','W2_W50_W51_W52_W53','W2_W50_W51_W52_W53_W54','W2_W50_W51_W52_W53_W54_W55','W2_W50_W51_W52_W53_W54_W55_W56','W2_W50_W51_W52_W53_W54_W55_W56_W57','W2_W50_W51_W52_W53_W54_W55_W56_W57_W58','W2_W50_W51_W52_W53_W54_W55_W56_W57_W58_W59',
# 'W2_W51','W2_W51_W52','W2_W51_W52_W53','W2_W51_W52_W53_W54','W2_W51_W52_W53_W54_W55','W2_W51_W52_W53_W54_W55_W56','W2_W51_W52_W53_W54_W55_W56_W57','W2_W51_W52_W53_W54_W55_W56_W57_W58','W2_W51_W52_W53_W54_W55_W56_W57_W58_W59',
# 'W2_W52','W2_W52_W53','W2_W52_W53_W54','W2_W52_W53_W54_W55','W2_W52_W53_W54_W55_W56','W2_W52_W53_W54_W55_W56_W57','W2_W52_W53_W54_W55_W56_W57_W58','W2_W52_W53_W54_W55_W56_W57_W58_W59',
# 'W2_W53','W2_W53_W54','W2_W53_W54_W55','W2_W53_W54_W55_W56','W2_W53_W54_W55_W56_W57','W2_W53_W54_W55_W56_W57_W58','W2_W53_W54_W55_W56_W57_W58_W59',
# 'W2_W54','W2_W54_W55','W2_W54_W55_W56','W2_W54_W55_W56_W57','W2_W54_W55_W56_W57_W58','W2_W54_W55_W56_W57_W58_W59',
# 'W2_W55','W2_W55_W56','W2_W55_W56_W57','W2_W55_W56_W57_W58','W2_W55_W56_W57_W58_W59',
# 'W2_W56','W2_W56_W57','W2_W56_W57_W58','W2_W56_W57_W58_W59',
# 'W2_W57','W2_W57_W58','W2_W57_W58_W59',
# 'W2_W58','W2_W58_W59'
# ]
# layerList = ['W2','W5']
