import gradio as gr
import os
from omegaconf import OmegaConf,ListConfig
import spaces
import torch

# from train import main as train_main
# from inference import inference as inference_main
import numpy as np

import transformers
# transformers.utils.move_cache()
from mycli import get_gt_img, generate_video

# @spaces.GPU()
def inference_model(
        target_image,
        prompt: str,
        emb_ckpt_path: str,
    ):
    
    gt_img = process_image(target_image)
    emb_ckpt_path = f'ckpts/FAE/motion_ckpts/{emb_ckpt_path}.pt'
    pretrained_model_name_or_path="cogvideox"
    return generate_video(
        gt_img=gt_img,
        prompt=prompt,
        emb_ckpt_path=emb_ckpt_path,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

def process_image(target_image):
    if target_image is None:
        return "No image provided."
    tensor_image = get_gt_img(target_image, 480, 720)
    return tensor_image


def get_checkpoints(checkpoint_dir):
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            checkpoints.append(file.split('.')[0])
    return checkpoints


def update_reference_video(checkpoint_dir):
    # get the parent dir of the checkpoint
    video_path = f"ckpts/FAE/reference_videos/{checkpoint_dir}.mp4"
    if not os.path.exists(video_path):
        print(f"Reference video missing: {video_path}")
    return gr.update(value=video_path)


def update_generated_prompt(text):
    return gr.update(value=text)


if __name__ == "__main__":

    if os.path.exists('results/custom'):
        os.system('rm -rf results/custom')
    if os.path.exists('outputs'):
        os.system('rm -rf outputs')

    inject_motion_embeddings_combinations = ['down 1280','up 1280','down 640','up 640']
    default_motion_embeddings_combinations = ['down 1280','up 1280']

    # Reference videos, Target Image, Prompt, Checkpoint
    examples_inference = [
        ####### crouch ####### 
        ['ckpts/FAE/reference_videos/crouch.mp4', 
         'benchmark/target_images/human/movie_man_31.webp', 
        '''A man is doing squats.''', 
         'crouch'],
         ####### Dance #######
        ['ckpts/FAE/reference_videos/Dance.mp4', 
         'benchmark/target_images/human/movie_man_32.webp', 
        '''A man is dancing.''', 
         'Dance'],
    ]

    gradio_theme = gr.themes.Default()
    with gr.Blocks(
        theme=gradio_theme,
        title="FlexiAct",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
            h1 {
                text-align: center;
                display: block;
            }
            h2 {
                text-align: center;
                display: block;
            }
            h3 {
                text-align: center;
                display: block;
            }
            .md_feedback li {
                margin-bottom: 0px !important;
            }
        """,
        head="""
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """,
    ) as demo:
        
        gr.Markdown(
            """
# FlexiAct: Towards Flexible Action Control in Heterogeneous Scenarios
<p align="center">
<a href="https://arxiv.org/abs/2403.20193"><img src='https://img.shields.io/badge/arXiv-2403.20193-b31b1b.svg'></a>
<a href=''><img src='https://img.shields.io/badge/Project_Page-MotionInversion(Coming soon)-blue'></a>
<a href='https://github.com/EnVision-Research/MotionInversion'><img src='https://img.shields.io/github/stars/EnVision-Research/MotionInversion?label=GitHub%20%E2%98%85&logo=github&color=C8C'></a>
<br>
<strong>Please consider starring <span style="color: orange">&#9733;</span> the <a href="https://github.com/EnVision-Research/MotionInversion" target="_blank" rel="noopener noreferrer">GitHub Repo</a> if you find this useful!</strong>
</p>
        """
        )
        with gr.Tabs(elem_classes=["tabs"]):
            with gr.Row():
                with gr.Column():
                    target_image = gr.Image(type="pil", label="Target Image")
                    text_input = gr.Textbox(label="Input Text")
                    checkpoint_dropdown = gr.Dropdown(label="Select Checkpoint", choices=get_checkpoints('ckpts/FAE/motion_ckpts'))
                
                with gr.Column():
                    reference_video = gr.Video(label="Reference Video")
                    output_video = gr.Video(label="Output Video")

            with gr.Row():
                inference_button = gr.Button("Generate Video")

        gr.Examples(
            examples=examples_inference,
            inputs=[reference_video,target_image, text_input, checkpoint_dropdown],
            outputs=output_video,
            examples_per_page=40,
        )
        checkpoint_dropdown.change(fn=update_reference_video, inputs=checkpoint_dropdown, outputs=reference_video)
        inference_button.click(inference_model, inputs=[target_image, text_input, checkpoint_dropdown], outputs=output_video)
        
        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=20410,
        )