import os
import sqlite3
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from PIL import UnidentifiedImageError
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import numpy as np

import numpy as np
from PIL import Image
from patchify import patchify
from transformers import CLIPProcessor, CLIPModel


#---------- Setting up the device --------------#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


Base_Dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


file = pd.read_excel(os.path.join(Base_Dir,"Dataset/Flickr_dataset/flickr30k_images/results.xlsx"))
all_imgs = os.listdir(os.path.join(Base_Dir,"Dataset/Flickr_dataset/flickr30k_images/flickr30k_images"))


#---------- Filtering out the image files --------------#
image_files = [f for f in all_imgs if f.endswith(('.jpg', '.png', '.jpeg'))]


conn = sqlite3.connect(os.path.join(Base_Dir,'Dataset/clip_reimagen_flicker.db'))
c = conn.cursor()


# Check if the table exists before dropping it
table_check_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='clip_reimagen_flicker';"
c.execute(table_check_query)
table_exists = c.fetchone()  # Returns None if the table does not exist

if table_exists:
    c.execute('''DROP TABLE clip_reimagen_flicker''')
    conn.commit()

# Create the table
c.execute('''CREATE TABLE clip_reimagen_flicker
             (img TEXT NOT NULL,
             text JSON NOT NULL,
             img_embed BLOB NOT NULL,
             top_text TEXT NOT NULL,
             clip_embed BLOB NOT NULL)''')

conn.commit()



#---------- Functions for image processing --------------#


# Define Image Preprocessing


def patchify_image(image, patch_height, patch_width):
    # Convert the image to a numpy array
    image_np = np.array(image)

    # Patchify the image

    patches = patchify(image_np, (patch_height, patch_width, 3), step=(patch_height,patch_width,3))
    # Reshape the patches to a list of images
    patches_reshaped = patches.reshape(-1, patch_height, patch_width, 3)

    return patches_reshaped





def get_num_channels(image):
    if isinstance(image, Image.Image):
        return len(image.getbands())
    else:
        return 0

def get_transform(image):
    num_channels = get_num_channels(image)
    # print(num_channels)
    transform_list = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
    if num_channels != 3:
        # print("Converting to 3 channels")
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transformer = transforms.Compose(transform_list)
    img_tensor = transformer(image)
    maxm,minm = img_tensor.max(),img_tensor.min()
    img_tensor = (img_tensor-minm)/(maxm - minm)
    return img_tensor.numpy().tobytes()





# Load a random 50K sample from LAION-400M
dir_path=os.path.join(Base_Dir,'Dataset/Flickr_dataset/flickr30k_images/flickr30k_images')

def create_table(fd,im_dataset):
    img_desc=[]

    for img in tqdm(im_dataset,desc="Creating Table"):
        row = file[file['image_name']==img]
        row = next(row.itertuples())
        curr_img = Image.open(os.path.join(dir_path,img)).resize((253,252))
        img_list = patchify_image(curr_img,36,23)
        clip_img_embed = []
        for clip_img in img_list:
            clip_img = Image.fromarray(clip_img)
            clip_embed = processor(images=clip_img, return_tensors="pt").to(device)
            # clip_embed = model.get_image_features(**clip_embed).cpu().detach().numpy()
            clip_embed = model.get_image_features(**clip_embed).cpu().detach().numpy()
            clip_img_embed.append(clip_embed[0])
        
        clip_img_embed = torch.tensor(clip_img_embed)
        clip_img_embed = clip_img_embed.unsqueeze(0)
        clip_embed = clip_img_embed.numpy().tobytes()
        curr_img_name = img
        img_embed = get_transform(curr_img)
        for _ in range(5):
            img_desc.append(row[3])
            row = next(file.itertuples())
        top_text = img_desc[0]
        img_file = json.dumps(img_desc)
        c.execute("INSERT INTO clip_reimagen_flicker (img,text,img_embed,top_text,clip_embed) VALUES (?,?,?,?,?)",(curr_img_name,img_file,img_embed,top_text,clip_embed))
        conn.commit()
        img_desc.clear()
    return


create_table(file,image_files)
print("Table created")
conn.close()






