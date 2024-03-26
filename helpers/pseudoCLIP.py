import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
#from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import clip
import open_clip

#model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

#model.encode_text("aaa")
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None):
    '''
        just copied from clip, references to clip.clip instead directly
        exists to build the text encoder with a pseudovector
    '''
    if name in clip.clip._MODELS:
        model_path = clip.clip._download(clip.clip._MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        if name in ["ViT-H/14", "ViT-h/14"]:
            model, _, preprocess = create_model_and_transforms("ViT-H/14", pretrained='laion2b_s32b_b79k')
            return model.to(device), preprocess
        elif name in ["ViT-G/14", "ViT-g/14"]:
            model, _, preprocess = create_model_and_transforms('ViT-g/14', pretrained='laion2b_s34b_b88k')
            return model.to(device), preprocess

        else:
            raise RuntimeError(f"Model {name} not found; available models = {clip.clip.available_models()} \n (or if it is grom open_clip add implementation in pseudoCLIP.load()")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                clip.clip.warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, clip.clip._transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.

        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, clip.clip._transform(model.input_resolution.item())
class CLIP_pseudo(clip.model.CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers,
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__(embed_dim=embed_dim, image_resolution=image_resolution, vision_layers=vision_layers,
                         vision_width=vision_width, vision_patch_size=vision_patch_size,
                         context_length=context_length, vocab_size=vocab_size, transformer_width=transformer_width,
                         transformer_heads=transformer_heads, transformer_layers=transformer_layers
                        )
    def encode_text(self, text, pseudowords=[], position_pseudo=2, print_intermediate=False):
        '''
        :param text: text embedding(s) that shall be modified by pseudowords
            number_text = {1, n} with number_pseudowords={1,n}
            --> 1->n =1 text with all pseudowords out,
            --> 1->n other way round,
            --> n->n  1-to-1-pairings of texts and pseudowords (text[i] with pseudoword[i])
        :param pseudowords: list/tensor of pseudowords, must be same dim as text embedding (e.g. 512 dim each)
            number_pseudowords={1,n}, see above
        :param position_pseudo: position to add the pseudowords
            example pseudo init as photo: "a photo of a cat" --> position_pseudo=2 as in (\start, a, photo, of, a, cat, \end)
                --> = "a <pseudoword> of a cat", use a known word like 'photo' so encoding is to a single token (NOT 'asfhgajfNS' or alike)
        :return:
        '''
        # text = [start token(49406), then word by word (subwords for unknown/madeup word), endtoken(49407)]
        if len(text.size()) > 2:
            #this may be a problem for tensors of size 1,1,n as one 1 should stay (but will not), but so far this should not occur
            text = torch.squeeze(text)

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if print_intermediate:
            print("token embedding", x, end="\n ======================== \n")

        #x = token_embedding same format as text but (e.g.)512 dim vector for each entry

        if len(pseudowords) != 0:
            if text.size(0) == 1:
                for n, pseudoword in enumerate(pseudowords):
                    x[0, position_pseudo] = pseudoword
                    if n == 0:
                        y = x.clone()
                    else:
                        y = torch.cat((y,x), 0)

            else:
                if len(pseudowords)==1:
                    for idx in range(x.size(0)):
                        x[idx, position_pseudo] = pseudowords[0]
                if len(pseudowords.size())==1:
                    for idx in range(x.size(0)): #if pseudowords are not shape (1, ...) but just (...)
                        x[idx, position_pseudo] = pseudowords
                if len(pseudowords)==x.size(0):
                    for idx in range(x.size(0)):
                        x[idx, position_pseudo] = pseudowords[idx]
                if len(pseudowords)!=1 and len(pseudowords)!=x.size(0) and len(pseudowords.size()) != 1:
                    exit("Error in combining texts and pseudowords: Numbers of pseudowords and texts do not fit: "
                         "must be either equal (both n) or one must be =1 (other arbitrary =n)")
        if print_intermediate:
            print("with pseudowords \n", y, end="\n ======================== \n")
        if len(pseudowords)!= 0 and text.size(0)==1:
            x = y + self.positional_embedding.type(self.dtype)
        else:
            x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_text_multiple_pseudo(self, text, pseudowords=[], positions_pseudo=[2]):
        '''
            encodes ONE set of pseudowords in the text
            -> len(pseudowords) = len(positions_pseudo)
        :param text: text embedding(s) that shall be modified by pseudowords

        :param pseudowords: list/tensor of pseudowords, must be same dim as text embedding (e.g. 512 dim each)
            number_pseudowords={1,n}, see above
        :param positions_pseudo: positions to add the pseudowords
            example pseudo init as photo: "a photo of a cat" --> position_pseudo=[2, 3] as in (\start, a, photo, of, a, cat, \end)
                --> = "a <pseudoword_1> <pseudoword_2> a cat", use known words like 'photo' so encodings are to a single token (NOT 'asfhgajfNS' or alike)
        :return:
        '''
        #print(pseudowords.size(), "\n ===", positions_pseudo)
        if len(pseudowords) != len(positions_pseudo):
            exit("PSEUDOCLIP ERROR: number of pseudowords and positions for them do not match")
        x = self.token_embedding(text).type(self.dtype)

        if len(pseudowords) != 0:
            for txt_idx in range(text.size(0)):
                for n, (pseudoword, position_pseudo) in enumerate(zip(pseudowords, positions_pseudo)):
                    x[txt_idx, position_pseudo] = pseudoword
        x = x+ self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP_pseudo(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    clip.model.convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()



####################################################################################
# for open_clipp
import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import numpy as np
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
from open_clip.coca_model import CoCa
from open_clip.loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from open_clip.openai import load_openai_model
from open_clip.pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from open_clip.transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from open_clip.tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs



def create_model_(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config = open_clip._get_hf_config(model_id, cache_dir)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
        model_cfg = config['model_cfg']
        pretrained_hf = False  # override, no need to load original HF text weights
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or open_clip.get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found; available models {open_clip.list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        if is_hf_model:
            # load pretrained weights for HF text model IFF no CLIP weights being loaded
            model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
        if custom_text:
            if "multimodal_cfg" in model_cfg:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype)
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = open_CLIP_pseudo(**model_cfg, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from open_clip.transformer import LayerNormFp32

                def _convert_ln(m):
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)
                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                open_clip.load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f' Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True
        elif has_hf_hub_prefix:
            logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
            open_clip.load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model



def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode)

    model = create_model_(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val



class open_CLIP_pseudo(open_clip.model.CLIP):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: open_clip.CLIPVisionCfg,
            text_cfg: open_clip.CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__(
            embed_dim= embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            init_logit_scale =init_logit_scale,
            init_logit_bias=init_logit_bias,
            cast_dtype=cast_dtype,
            output_dict=output_dict
        )
        self.dtype = torch.float32
    def encode_text(self, text, pseudowords=[], position_pseudo=2, print_intermediate=False):
        '''
        :param text: text embedding(s) that shall be modified by pseudowords
            number_text = {1, n} with number_pseudowords={1,n}
            --> 1->n =1 text with all pseudowords out,
            --> 1->n other way round,
            --> n->n  1-to-1-pairings of texts and pseudowords (text[i] with pseudoword[i])
        :param pseudowords: list/tensor of pseudowords, must be same dim as text embedding (e.g. 512 dim each)
            number_pseudowords={1,n}, see above
        :param position_pseudo: position to add the pseudowords
            example pseudo init as photo: "a photo of a cat" --> position_pseudo=2 as in (\start, a, photo, of, a, cat, \end)
                --> = "a <pseudoword> of a cat", use a known word like 'photo' so encoding is to a single token (NOT 'asfhgajfNS' or alike)
        :return:
        '''
        # text = [start token(49406), then word by word (subwords for unknown/madeup word), endtoken(49407)]
        if len(text.size()) > 2:
            #this may be a problem for tensors of size 1,1,n as one 1 should stay (but will not), but so far this should not occur
            text = torch.squeeze(text)

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if print_intermediate:
            print("token embedding", x, end="\n ======================== \n")

        #x = token_embedding same format as text but (e.g.)512 dim vector for each entry

        if len(pseudowords) != 0:
            if text.size(0) == 1:
                for n, pseudoword in enumerate(pseudowords):
                    x[0, position_pseudo] = pseudoword
                    if n == 0:
                        y = x.clone()
                    else:
                        y = torch.cat((y,x), 0)

            else:
                if len(pseudowords)==1:
                    for idx in range(x.size(0)):
                        x[idx, position_pseudo] = pseudowords[0]
                if len(pseudowords.size())==1:
                    for idx in range(x.size(0)): #if pseudowords are not shape (1, ...) but just (...)
                        x[idx, position_pseudo] = pseudowords
                if len(pseudowords)==x.size(0):
                    for idx in range(x.size(0)):
                        x[idx, position_pseudo] = pseudowords[idx]
                if len(pseudowords)!=1 and len(pseudowords)!=x.size(0) and len(pseudowords.size()) != 1:
                    exit("Error in combining texts and pseudowords: Numbers of pseudowords and texts do not fit: "
                         "must be either equal (both n) or one must be =1 (other arbitrary =n)")
        if print_intermediate:
            print("with pseudowords \n", y, end="\n ======================== \n")
        if len(pseudowords)!= 0 and text.size(0)==1:
            x = y + self.positional_embedding.type(self.dtype)
        else:
            x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_text_multiple_pseudo(self, text, pseudowords=[], positions_pseudo=[2]):
        '''
            encodes ONE set of pseudowords in the text
            -> len(pseudowords) = len(positions_pseudo)
        :param text: text embedding(s) that shall be modified by pseudowords

        :param pseudowords: list/tensor of pseudowords, must be same dim as text embedding (e.g. 512 dim each)
            number_pseudowords={1,n}, see above
        :param positions_pseudo: positions to add the pseudowords
            example pseudo init as photo: "a photo of a cat" --> position_pseudo=[2, 3] as in (\start, a, photo, of, a, cat, \end)
                --> = "a <pseudoword_1> <pseudoword_2> a cat", use known words like 'photo' so encodings are to a single token (NOT 'asfhgajfNS' or alike)
        :return:
        '''
        #print(pseudowords.size(), "\n ===", positions_pseudo)
        if len(pseudowords) != len(positions_pseudo):
            exit("PSEUDOCLIP ERROR: number of pseudowords and positions for them do not match")
        x = self.token_embedding(text).type(self.dtype)

        if len(pseudowords) != 0:
            for txt_idx in range(text.size(0)):
                for n, (pseudoword, position_pseudo) in enumerate(zip(pseudowords, positions_pseudo)):
                    x[txt_idx, position_pseudo] = pseudoword
        x = x+ self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


