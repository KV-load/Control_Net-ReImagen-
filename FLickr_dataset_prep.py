import os
import sqlite3
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from PIL import UnidentifiedImageError
import pandas as pd


Base_Dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


file = pd.read_excel(os.path.join(Base_Dir,"Dataset/Flickr_dataset/flickr30k_images/results.xlsx"))
all_imgs = os.listdir(os.path.join(Base_Dir,"Dataset/Flickr_dataset/flickr30k_images/flickr30k_images"))


#---------- Filtering out the image files --------------#
image_files = [f for f in all_imgs if f.endswith(('.jpg', '.png', '.jpeg'))]


conn = sqlite3.connect(os.path.join(Base_Dir,'Dataset/reimagen_flicker.db'))
c = conn.cursor()


# Check if the table exists before dropping it
table_check_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='reimagen_flicker';"
c.execute(table_check_query)
table_exists = c.fetchone()  # Returns None if the table does not exist

if table_exists:
    c.execute('''DROP TABLE reimagen_flicker''')
    conn.commit()

# Create the table
c.execute('''CREATE TABLE reimagen_flicker
             (img TEXT NOT NULL,
             text JSON NOT NULL,
             img_embed BLOB NOT NULL,
             top_text TEXT NOT NULL)''')

conn.commit()



#---------- Setting up the device --------------#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#---------- Functions for image processing --------------#


# Define Image Preprocessing
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
        curr_img = Image.open(os.path.join(dir_path,img))
        curr_img_name = img
        img_embed = get_transform(curr_img)
        for _ in range(5):
            img_desc.append(row[3])
            row = next(file.itertuples())
        top_text = img_desc[0]
        img_file = json.dumps(img_desc)
        c.execute("INSERT INTO reimagen_flicker (img,text,img_embed,top_text) VALUES (?,?,?,?)",(curr_img_name,img_file,img_embed,top_text))
        conn.commit()
        img_desc.clear()
    return


create_table(file,image_files)
print("Table created")
conn.close()






