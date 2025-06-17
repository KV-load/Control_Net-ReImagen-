from diffusers import AutoencoderKL, StableDiffusionPipeline
import torch
import os
import sqlite3
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import scann
import torch.nn.functional as F
from torch import optim
from Re_Imagen_train.control_net_unet import ReImagen
import gradio as gr
import cv2
from ControlNet.annotator.util import resize_image, HWC3
from ControlNet.annotator.uniformer import UniformerDetector

#-------------- device config ---------------------------------#
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # for reducing memory fragmentation
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#------------------- loading the model ---------------------#
ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Checkpoints/Control_Net_arch5.ckpt'))

# Load models only once to save memory and improve efficiency
sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
org_unet = sd_pipeline.unet.to(device)

model = ReImagen(org_unet,False)
model.load_state_dict(torch.load(ckpt_dir))
model.eval()

# ------------------ loading necessary components of the model ---------------------#
# Use components from the already loaded pipeline
text_tokenizer = sd_pipeline.tokenizer
vae = sd_pipeline.vae.to(device)
scheduler = sd_pipeline.scheduler
text_encoder = sd_pipeline.text_encoder.to(device)

# ------------------ loading the dataset ---------------------#
Base_Dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
Dataset_dir = os.path.join(Base_Dir, 'Dataset/clip_reimagen_flicker.db')
conn = sqlite3.connect(Dataset_dir)
c = conn.cursor()

c.execute("UPDATE clip_reimagen_flicker SET top_text=' ' WHERE text IS NULL")
conn.commit()

text_dataset = c.execute("SELECT top_text FROM clip_reimagen_flicker").fetchall()
text_dataset = [i[0] for i in text_dataset]

img_dataset = c.execute("SELECT img_embed FROM clip_reimagen_flicker").fetchall()
img_dataset = [i[0] for i in img_dataset]

img_png = c.execute("SELECT img FROM clip_reimagen_flicker").fetchall()
img_png = [i[0] for i in img_png]

clip_embed = c.execute("SELECT clip_embed FROM clip_reimagen_flicker").fetchall()
clip_embed = [i[0] for i in clip_embed]


# --------------------------- initiating the searcher ---------------------#
# Ensure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

vectorizer = TfidfVectorizer()

# Tokenize the alt-texts
tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in text_dataset]
vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_corpus])

# Build ScaNN searcher
searcher = scann.scann_ops_pybind.load_searcher(os.path.join(Base_Dir,'Dataset/clip_scann_index/'))


#-----------------Prepare the image for the model---------------#
transformer = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

apply_uniformer = UniformerDetector(device)

def seg2image(image):
    image = np.array(image)
    input_image = HWC3(image)
    detected_map = apply_uniformer(resize_image(input_image, 1080))
    img = resize_image(input_image, 720)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(detected_map)



# ------------------ necessary functions ---------------------#
@torch.no_grad()
def similar(searcher, query):
    """Find similar images based on text query."""
    tokenized_query = nltk.word_tokenize(query.lower())
    query_embedding = vectorizer.transform([' '.join(tokenized_query)]).toarray()

    # Search for the nearest neighbors
    neighbors, distances = searcher.search_batched(query_embedding, final_num_neighbors=5)

    # Retrieve top N results
    top_n = min(4, len(neighbors[0]))  # Ensure we don't exceed available results
    top_n_indices = neighbors[0][:top_n]

    similar_images = []
    similar_img_png = []
    similar_img_clip_embed = []
    
    for i in top_n_indices:
        if i < len(img_dataset) and i < len(clip_embed) and i < len(img_png):
            img = torch.from_numpy(np.frombuffer(img_dataset[i], dtype=np.float32).reshape(3, 64, 64))
            similar_img_clip_embed.append(torch.from_numpy(np.frombuffer(clip_embed[i], dtype=np.float32).reshape(1, 77, 768)))
            
            print(f"img_dataset[{i}] shape: {np.frombuffer(img_dataset[i], dtype=np.float32).shape}")
            print(f"clip_embed[{i}] shape: {np.frombuffer(clip_embed[i], dtype=np.float32).shape}")
            
            print(img_png[i])
            img_path = os.path.join(Base_Dir, "Dataset/Flickr_dataset/flickr30k_images/flickr30k_images/", img_png[i])
            if os.path.exists(img_path):
                similar_images.append(img)
                similar_img_png.append(Image.open(img_path))
    
    # Ensure we have enough results for indexing
    if len(similar_images) < 2 or len(similar_img_png) < 2 or len(similar_img_clip_embed) < 2:
        print("Warning: Not enough similar images found. Using available ones.")
        # Duplicate the last item if needed
        while len(similar_images) < 2:
            similar_images.append(similar_images[-1] if similar_images else torch.zeros(3, 64, 64))
        while len(similar_img_png) < 2:
            similar_img_png.append(similar_img_png[-1] if similar_img_png else Image.new('RGB', (64, 64)))
        while len(similar_img_clip_embed) < 2:
            similar_img_clip_embed.append(similar_img_clip_embed[-1] if similar_img_clip_embed else torch.zeros(1, 77, 768))
    
    return similar_img_clip_embed, similar_images[0:2], similar_img_png[0:2],similar_img_png


def sample(ref_img, query, text_guidance, img_guidance,alpha=0.5):
    params_dict = dict(model.named_parameters())
    scheduler.set_timesteps(50) # Use fewer steps for faster inference

    # Extracting null embeddings from the model parameters
    # for name, _ in model.named_parameters():
    #     if 'null' in name:
    #         print(name)
    #null embeddings 
    img_null = params_dict["img_null_entry"]
    txt_null = params_dict["txt_null_entry"]
    clip_null = params_dict["clip_null_entry"]


     # Preparing the similar images
    sim_img_clip_embed, similar_imgs, sim_img_png,all_imgs = similar(searcher, query)
    demo =  gr.Interface(fn=None,inputs=None,outputs=gr.Gallery(value=all_imgs, label="Similar Images"))       
    demo.launch(server_name="10.21.226.130",server_port=7860)


   #getting the refrence image
    ref_img = all_imgs[0]
    ref_img = transformer(ref_img).unsqueeze(0).to(device)
    # t = scheduler.timesteps[0]  # Use the *first timestep* of your schedule (e.g., t ~980 for 60-step)
    ref_img = scheduler.add_noise(ref_img, torch.randn_like(ref_img), torch.tensor([100], device=device))
    with torch.no_grad():
        ref_img = vae.encode(ref_img * 2 - 1)
        ref_img = 0.18215 * ref_img.latent_dist.sample() 

    # Convert reference image to tensor and move to device
    base_img = all_imgs[1]
    base_img = seg2image(base_img)
    base_img = transformer(base_img).unsqueeze(0).to(device)
    
    # Encode to latent space
    with torch.no_grad():
            base_img = vae.encode(base_img * 2 - 1)
            base_img = 0.18215 * base_img.latent_dist.sample()
    
   
    # Prepare the text input
    text_query_emb = text_tokenizer(
        query,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_query_emb = text_encoder(text_query_emb.input_ids)[0]
    
    # Setting up diffusion process
    if not torch.isfinite(ref_img).all():
        print("Warning: Non-finite values detected in ref_img!")
    # print("ref_img stats:", ref_img.min().item(), ref_img.max().item(), ref_img.mean().item())


    # Preparing the noisy image
    # t = torch.tensor([999], device=device)
    # noise = torch.randn_like(base_img)
    # noisy_img = scheduler.add_noise(ref_img, noise, t)

    # base_img = Image.new('RGB', (512, 512))
    # base_img = transformer(base_img).unsqueeze(0)
    # base_img = base_img.to(device)
    # base_img = scheduler.add_noise(base_img, noise, t)
    # base_img = vae.encode(base_img).latent_dist.sample()*0.18215
   
    # Add noise to the reference image latent
    # ref_img = torch.rand_like(ref_img)
    # Denoising loop
    similar_imgs_modified = [transformer(i).unsqueeze(0).to(device) for i in sim_img_png]
    for idx,sim_img in enumerate(similar_imgs_modified):
            sim_img = vae.encode(sim_img * 2 - 1)
            sim_img = 0.18215 * sim_img.latent_dist.sample()
            similar_imgs_modified[idx] = sim_img

    for i in tqdm(reversed(scheduler.timesteps - 1)):
        i_tensor = torch.tensor([i], device=device)
        # print(i)
        # Convert and prepare similar image
        # with torch.no_grad():
            # Use VAE to encode the similar image for consistent processing
            # sim_latent = vae.encode(sim_img_tensors[0]).latent_dist.sample()
        # print("ref_img stats:", ref_img.min().item(), ref_img.max().item(), ref_img.mean().item())
        
        with torch.inference_mode():    
            # Get model prediction
            noise_pred = model(ref_img,i_tensor,text_query_emb,base_img,sim_img_clip_embed,similar_imgs_modified)

            # for only image guidance
            noise_pred_ntxt = model(ref_img,i_tensor,txt_null,base_img,sim_img_clip_embed,similar_imgs_modified)

            # for only text guidance
            similar_imgs_modified_nimg = [img_null for _ in range(2)]
            sim_img_clip_embed_nimg = [clip_null for _ in range(2)]
            noise_pred_nimg = model(ref_img,i_tensor,text_query_emb,base_img,sim_img_clip_embed_nimg,similar_imgs_modified_nimg)


            # Net prediction
            img_pred = noise_pred * img_guidance - noise_pred_nimg * (img_guidance - 1)
            txt_pred = noise_pred * text_guidance - noise_pred_ntxt * (text_guidance - 1)

            noise_pred = txt_pred + img_pred 

            # noise_max = noise_pred.max()
            # noise_min = noise_pred.min()
            # Normalize the noise prediction
            # if noise_max > 1 or noise_min < -1:
                # noise_pred =4* (noise_pred - noise_min) / (noise_max - noise_min) * 2 - 1
                # print("Normalized noise_pred to range [-1, 1]")

            print("noise_pred stats:", noise_pred.min().item(), noise_pred.max().item(), noise_pred.mean().item())
            # noise_pred = noise_pred.clamp(-1, 1)  # Ensure the prediction is within valid range

        # print(noise_pred)
        # loading all on one device
        noise_pred = noise_pred.to(device)
        # noisy_img = noisy_img.detach().to(device)
        ref_img = ref_img.to(device)
        # print(noise_pred)
        # Compute the previous noisy sample x_t -> x_t-1
        ref_img = scheduler.step(noise_pred, i, ref_img).prev_sample
        saved_img = ref_img
        # Decode the image
        with torch.no_grad():
            saved_img = saved_img / 0.18215
            saved_img = vae.decode(saved_img).sample
        
        # Post-processing
        # saved_img = saved_img * 2 - 1
        saved_img = ((saved_img+1)/2).clamp(0, 1)
        saved_img = saved_img.cpu().permute(0, 2, 3, 1).numpy()
        saved_img = (saved_img * 255).astype(np.uint8)
        saved_img = Image.fromarray(saved_img[0])
        
        output_path = os.path.join(Base_Dir, "Re_Imagen_demo/default/samples")
        saved_img.save(os.path.join(output_path, f"{i}.png"))
        # print(f"Generated image saved to {output_path}")
    
    return ref_img


# Test the sample function
sample_img_path = os.path.join(Base_Dir, "Dataset/Flickr_dataset/flickr30k_images/flickr30k_images/3397803103.jpg")
if os.path.exists(sample_img_path):
    sample_img = Image.open(sample_img_path)
    text_guidance = 7
    img_guidance = 3

    result = sample(sample_img, "The dog playing with ball ",text_guidance, img_guidance)
else:
    print(f"Error: Sample image not found at {sample_img_path}")
