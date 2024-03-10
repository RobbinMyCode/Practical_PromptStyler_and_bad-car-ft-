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
        raise RuntimeError(f"Model {name} not found; available models = {clip.clip.available_models()}")

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

