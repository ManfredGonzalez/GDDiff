"""LDFA (official notebook code, ldfa/run_ldfa.py) on a folder: python run_ldfa_folder.py <in> <out>.
Same settings except padding_mask_crop (rejected for the 9-channel inpainting UNet by this diffusers version;
it only crops the region before the identical 512x512, 50-step sampling). SD2-inpainting loaded from the sd2-community mirror (stabilityai repo removed from the Hub)."""
import os
import sys

LDFA = os.path.expanduser('~/Desktop/master_research/ldfa')
sys.path.insert(0, LDFA)
os.chdir(LDFA)
import torch  # noqa: E402
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline  # noqa: E402
from face_parsing.face_parse import get_face_parsing, get_face_parsing_model  # noqa: E402
from PIL import Image  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
segmentation_model = get_face_parsing_model('segmentation_model/79999_iter.pth')
pipeline = StableDiffusionInpaintPipeline.from_pretrained('sd2-community/stable-diffusion-2-inpainting',
                                                          torch_dtype=torch.float16, variant='fp16').to('cuda')
pipeline.scheduler = EulerAncestralDiscreteScheduler()
rng = torch.Generator()
rng.manual_seed(0)
for f in sorted(os.listdir(src)):
    image = Image.open(os.path.join(src, f))
    mask = get_face_parsing([image], model=segmentation_model)[0]
    mask[mask == 14] = 0
    mask[mask == 16] = 0
    mask[mask == 17] = 0
    mask[mask >= 1] = 255
    mask = Image.fromarray(mask).resize((512, 512), Image.NEAREST)
    out = pipeline(prompt='', image=image.resize((512, 512)), mask_image=mask, guidance_scale=1,
                   generator=rng, num_inference_steps=50).images[0]
    out.save(os.path.join(dst, f))
