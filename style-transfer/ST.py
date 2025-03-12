import os
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from os.path import basename, splitext
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import models.transformer as transformer
import models.StyTR as StyTR
from argparse import Namespace

def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(content_dir, style_dir, output_dir):
    vgg_path = "./experiments/vgg_normalised.pth"
    decoder_path = "./experiments/decoder_iter_160000.pth"
    trans_path = "./experiments/transformer_iter_160000.pth"
    embedding_path = "./experiments/embedding_iter_160000.pth"
    args = Namespace(
        content=content_dir,  # Not used when content_dir is provided
        content_dir=None,
        style=style_dir,  # Not used when style_dir is provided
        style_dir=None,
        output=output_dir,
        vgg=vgg_path,
        decoder_path=decoder_path,
        Trans_path=trans_path,
        embedding_path=embedding_path,
        style_interpolation_weights="",  # Default value
        a=1.0,  # Default value
        position_embedding='sine',  # Default value
        hidden_dim=512,  # Default value
    )
    # Advanced options
    content_size=512
    style_size=512
    crop='store_true'
    save_ext='.jpg'
    output_path=args.output
    preserve_color='store_true'
    alpha=args.a


    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Either --content or --content_dir should be given.
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --style_dir should be given.
    if args.style:
        style_paths = [Path(args.style)]    
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    if not os.path.exists(output_path):
        os.mkdir(output_path)


    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(args.decoder_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.Trans_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.embedding_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
    network.eval()
    network.to(device)



    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    for content_path in content_paths:
        for style_path in style_paths:
            print(content_path)
        
            content_tf1 = content_transform()       
            content = content_tf(Image.open(content_path).convert("RGB"))

            h, w, c = np.shape(content)    
            style_tf1 = style_transform(h, w)
            style = style_tf(Image.open(style_path).convert("RGB"))

            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            
            with torch.no_grad():
                output = network(content, style)  # output is a tuple
                print(output)  # Inspect the structure of the output
                output = output[0]  # Extract the first element (stylized image)
            
            output = output.cpu()  # Move the tensor to CPU
                    
            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                output_path, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], save_ext
            )
    
            save_image(output, output_name)



