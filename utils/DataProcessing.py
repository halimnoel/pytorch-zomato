import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def generte_label(restaurant_cuisine, cuisines):
	label = torch.zeros(70)
	for rc in restaurant_cuisine:
		i = cuisines.index(rc)
		label[i] = 1
	return label

class ImageDataset(Dataset):
	def __init__(self, image_base_dir, json_file, cuisines):
		self.data = []
		self.labels = []
		file_restaurant = open(json_file)
		restaurant_data = json.load(file_restaurant)
		for rd in restaurant_data:
			image_dir = rd['res_link'].split('/')[-1]
			label = generte_label(rd['cuisines'], cuisines)
			for image_file in os.listdir(image_base_dir+'/'+image_dir):
				try:
					im = Image.open(image_base_dir+'/'+image_dir+'/'+image_file)
					im = im.resize((224, 224))
					image_array = np.asarray(im)
					if image_array.shape[2] == 3:
						self.data.append(image_array)
						self.labels.append(label)
				except: continue

	def __len__(self): return len(self.data)
	def __getitem__(self, idx):
		data = self.data[idx]
		label = self.labels[idx]
		return (transform(data), label)