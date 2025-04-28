import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

with open("../hf_access_tokens.txt", "r") as f:
    token = f.readlines()[0].strip()

login(
    token
)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "../assets/ckpts/uni2-h/"
os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
hf_hub_download(
    "MahmoodLab/UNI2-h",
    filename="pytorch_model.bin",
    local_dir=local_dir,
    force_download=True,
)
timm_kwargs = {
    "model_name": "vit_giant_patch14_224",
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}
model = timm.create_model(pretrained=False, **timm_kwargs)
model.load_state_dict(
    torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"),
    strict=True,
)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model.eval()
from PIL import Image

image = Image.open("./crops/slide_example/24864_21728.png")
image = transform(image).unsqueeze(
    dim=0
)  # Image (torch.Tensor) with shape [1, 3, 224, 224] following image resizing and normalization (ImageNet parameters)
with torch.inference_mode():
    feature_emb = model(image)
print(feature_emb)
