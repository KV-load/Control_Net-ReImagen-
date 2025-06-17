from diffusers import StableDiffusionPipeline, AutoencoderKL
import torch
import sqlite3
from PIL import Image,ImageOps
import nltk
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import scann
from control_net_unet import ReImagen
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torchvision import transforms
import cv2
from ControlNet.annotator.util import resize_image, HWC3
from ControlNet.annotator.uniformer import UniformerDetector
from torch.utils.data import Dataset,DataLoader
from copy import deepcopy
from torch.amp import autocast, GradScaler
import random
from time import sleep, time


org_unet_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ctrl_unet_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for reducing memory fragmentation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Base_Dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


#-------------- Loading Dataset ---------------------------------#

ckpt_dir = os.path.join(Base_Dir,'Checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, 'Control_Net_arch5.ckpt')

# Load the COYO-700M dataset with streaming
conn = sqlite3.connect(os.path.join(Base_Dir,'Dataset/clip_reimagen_flicker.db'))
c = conn.cursor()
c.execute("UPDATE clip_reimagen_flicker SET top_text=' ' WHERE text IS NULL")
conn.commit()

text_dataset =  c.execute("SELECT top_text FROM clip_reimagen_flicker")
text_dataset = [i[0] for i in text_dataset]
# print(text_dataset)

img_dataset = c.execute("SELECT img_embed FROM clip_reimagen_flicker")
img_dataset = [i[0] for i in img_dataset]
# Take a small subset of the dataset (e.g., first 20 samples)

img_png= c.execute("SELECT img FROM clip_reimagen_flicker")
img_png = [i[0] for i in img_png]

clip_embed = c.execute("SELECT clip_embed FROM clip_reimagen_flicker")
clip_embed = [i[0] for i in clip_embed]

vectorizer = TfidfVectorizer()

# Tokenize the alt-texts
tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in text_dataset]
vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_corpus])

# Build ScaNN searcher
searcher = scann.scann_ops_pybind.load_searcher(os.path.join(Base_Dir,'Dataset/clip_scann_index/'))

#Preprocessing the image
transformer = transforms.Compose([transforms.Resize((512,512)),
                                  transforms.ToTensor()])





# ----------------- Creating custom dataset class -----------------#
class CustomDataset(Dataset):
    def __init__(self, img_dataset, text_dataset):
        self.img_dataset = img_dataset
        self.text_dataset = text_dataset

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
         return self.img_dataset[idx], self.text_dataset[idx]




# creating a probab function to get the inference guidance scale
def probab(chance=0.1):
    return random.random() >= chance



@torch.no_grad()
def similar(searcher, query):
    """Find similar images based on text query."""
    tokenized_query = nltk.word_tokenize(query.lower())
    query_embedding = vectorizer.transform([' '.join(tokenized_query)]).toarray()

    # Search for the nearest neighbors
    neighbors, distances = searcher.search_batched(query_embedding, final_num_neighbors=5)

    # Retrieve top N results
    top_n = 4
    top_n_indices = neighbors[0][:top_n]

    similar_images = []
    similar_img_png = []
    similar_img_clip_embed = []
    for i in top_n_indices:
        img = torch.from_numpy(np.frombuffer(img_dataset[i], dtype=np.float32).copy().reshape(3,64,64)) # converting from BLOB to tensor
        similar_img_clip_embed.append(torch.from_numpy(np.frombuffer(clip_embed[i], dtype=np.float32).copy().reshape(1,77,768))) # converting from BLOB to tensor
        if img is not None:
            similar_images.append(img)
            similar_img_png.append(Image.open(Base_Dir+"/Dataset/Flickr_dataset/flickr30k_images/flickr30k_images/"+img_png[i]))

    return similar_img_clip_embed,similar_images[1:3] , similar_img_png[1:3]  # Limit to 3 images


    
          
#----------------- Preparing similar images for the model ---------------#
def canny(image):
    """Apply Canny edge detection to the image."""
    image = ImageOps.grayscale(image)
    image = cv2.Canny(np.array(image), 100, 200)
    edges = Image.fromarray(image)
    return edges 

apply_uniformer = UniformerDetector(device) #initialising the uniformer detector

def seg2image(image):
    image = np.array(image)
    input_image = HWC3(image)
    detected_map = apply_uniformer(resize_image(input_image, 1080))
    img = resize_image(input_image, 720)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(detected_map)





org_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)

org_unet = org_pipe.unet


#Specifying sizes of the inp and output of downblocks


model = ReImagen(org_unet)
model.train()

criterian = torch.nn.MSELoss()
optimiser = optim.Adam(model.get_params(),lr=1e-5)
num_epochs = 10


#loading essential components of the model
scheduler = org_pipe.scheduler
text_tokeniser = org_pipe.tokenizer
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="vae").to(device)
text_encoder = org_pipe.text_encoder.to(device)



#----------------- Loading checkpoint if it exists ---------------------#
if os.path.exists(ckpt_path):
    print("Loading checkpoint...")
    model.load_state_dict(torch.load(ckpt_path))

    # checking loading of parameters is done on the correct device
    # Verify device placement
    assert next(model.org_unet.parameters()).device == org_unet_device
    assert next(model.cpy_unet.parameters()).device == ctrl_unet_device
else:
    print("No checkpoint found. Starting training from scratch.")




#---------Reassuring table is created and data is loaded in it---------#
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = c.fetchall()
print("Tables in the database:", tables)
# Count the total number of rows for tqdm progress bar
c.execute("SELECT COUNT(*) FROM clip_reimagen_flicker")
total_rows = c.fetchone()[0]

loss_epoch = []

#---------------- Preparing the dataset---------------#
c.execute("SELECT * FROM clip_reimagen_flicker")
all_rows = c.fetchall()
conn.close


# initiating the batch size
batch_size = 1
train_dataset = CustomDataset(img_dataset, text_dataset)
train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Training optimisers
# scaler = GradScaler(device=device)


#---------------- Waiting for the gpu to be free---------#
def wait_for_gpu(min_free_memory=1024):  # min_free_memory in MB
    while True:
        torch.cuda.empty_cache()
        total = torch.cuda.get_device_properties(0).total_memory
        used = torch.cuda.memory_allocated(0)
        free = (total - used) / (1024 ** 2)  # in MB
        if free >= min_free_memory:
            break
        time.sleep(5)

print("Null Embedding Parameters:")
for name, param in model.named_parameters():
    if "null_entry" in name:
        print(f"{name}:")
        print(f"  Shape: {tuple(param.shape)}")
        print(f"  Requires grad: {param.requires_grad}")
        print(f"  Data type: {param.dtype}")
        print(f"  Device: {param.device}\n")

#----------- Setting up dummmy inputs for the model when text and image conditions are not provided -----------#
dummy_text = torch.zeros((1, 77, 768), device=device)  # Dummy text embedding
dummy_similar_images = [torch.zeros((1, 4, 64, 64), device=device) for _ in range(2)]  # Dummy similar images
dummy_clip_embed = [torch.zeros((1, 77, 768), device=device) for _ in range(2)]  # Dummy CLIP embeddings

# setting timesteps
scheduler.set_timesteps(1000, device=device)
# Training loop
for epoch_idx in range(num_epochs):
    
    losses=[]

    np.random.shuffle(all_rows)
    for steps, row in enumerate(tqdm(all_rows, total=total_rows, desc="Epoch {}/{}".format(epoch_idx + 1, num_epochs))):

        img,img_desc,_,text_query,_ = row

        # creating image latent

    
        ref_img =Image.open(Base_Dir+'/Dataset/Flickr_dataset/flickr30k_images/flickr30k_images/'+img)
        base_img = deepcopy(ref_img)
        
        ref_img = transformer(ref_img).unsqueeze(0).to(device)

        # ref_img = torch.stack(ref_img)
        # print(ref_img.shape)
        with torch.no_grad():
            ref_img = vae.encode(ref_img * 2 - 1)
            ref_img = 0.18215 * ref_img.latent_dist.sample() #Stable Diffusion's latents need scaling by a factor (0.18215).
            
            
            
            

        # creating text embedding and tokenising it
        text_query_emb = text_tokeniser(
        text_query,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
        ).to(device)

        text_query_emb = text_encoder(text_query_emb.input_ids)[0]

        # finding similar images we get
        simlr_img_clip_embed,_,similar_images = similar(searcher, text_query)
        #creating noisy image
        noise = torch.randn_like(ref_img[0])

        t = torch.randint(0, 1000, (ref_img.shape[0],)).to(device)

        noisy_img = scheduler.add_noise(ref_img,noise,t) 


        # getting noise from the model
        t = t.float()

        similar_images_segmentation= [seg2image(i) for i in similar_images[:2]]
        base_img = seg2image(base_img)
        base_img = transformer(base_img).unsqueeze(0).to(device)

        # print(base_img.shape)  # converting to tensor and adding batch dimension
        similar_imgs_modified = [transformer(i).unsqueeze(0).to(device) for i in similar_images[:2]]

        #----------preparing similar images for the process----------#
        for idx,sim_img in enumerate(similar_imgs_modified):
            sim_img = vae.encode(sim_img * 2 - 1)
            sim_img = 0.18215 * sim_img.latent_dist.sample()
            similar_imgs_modified[idx] = sim_img


        simlr_img_clip_embed = [i for i in simlr_img_clip_embed]

        # print(torch.cuda.memory_summary())
    
        # preparing refrence image and similar images for the model
        with torch.no_grad():
            base_img = vae.encode(base_img * 2 - 1)
            base_img = 0.18215 * base_img.latent_dist.sample()



        # text and image free guidance
        # with autocast(device_type='cuda'):

        try:   
            pred_noise = model(noisy_img,t,text_query_emb,base_img,simlr_img_clip_embed,similar_imgs_modified)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory error. Waiting for GPU to free up...")
                wait_for_gpu(min_free_memory=1024)
                print("Retrying...")
                pred_noise = model(noisy_img, t, text_query_emb, base_img, simlr_img_clip_embed, similar_imgs_modified)
            else:
                raise e
        
        # Backpropagation for model
        noise = noise.unsqueeze(0).to(device) # as pred_noise is of shape (1,4,64,64) and noise is of shape (4,64,64)
        pred_noise = pred_noise.to(device)# print(pred_noise)
        loss = criterian(noise,pred_noise)
#     scaler.scale(loss).backward()
        
        losses.append(loss.item())
        # scaler.scale(loss).backward()  # Use scaler to scale the loss for mixed precision training
        # scaler.step(optimiser)
        # scaler.update()
        # optimiser.zero_grad(set_to_none=True) 
        loss.backward() 
        optimiser.step()
        optimiser.zero_grad()

        del noisy_img, ref_img, similar_imgs_modified, similar_images_segmentation, simlr_img_clip_embed, text_query_emb, pred_noise, noise
        # Clear GPU memory
        if steps % 100 == 0:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
   

    # Log and save model
    # print(losses)
    epoch_loss = np.mean(losses)
    print(f"Epoch {epoch_idx + 1}: Loss = {epoch_loss:.4f}")
    loss_epoch.append(epoch_loss)
    torch.save(model.state_dict(), ckpt_path)


print("Losses/epoch:",loss_epoch)
print('Training complete.')







