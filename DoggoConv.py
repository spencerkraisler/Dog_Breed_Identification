import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# 120 dogbreeds
breeds_list = "affenpinscher,afghan_hound,african_hunting_dog,airedale,american_staffordshire_terrier,appenzeller,australian_terrier,basenji,basset,beagle,bedlington_terrier,bernese_mountain_dog,black-and-tan_coonhound,blenheim_spaniel,bloodhound,bluetick,border_collie,border_terrier,borzoi,boston_bull,bouvier_des_flandres,boxer,brabancon_griffon,briard,brittany_spaniel,bull_mastiff,cairn,cardigan,chesapeake_bay_retriever,chihuahua,chow,clumber,cocker_spaniel,collie,curly-coated_retriever,dandie_dinmont,dhole,dingo,doberman,english_foxhound,english_setter,english_springer,entlebucher,eskimo_dog,flat-coated_retriever,french_bulldog,german_shepherd,german_short-haired_pointer,giant_schnauzer,golden_retriever,gordon_setter,great_dane,great_pyrenees,greater_swiss_mountain_dog,groenendael,ibizan_hound,irish_setter,irish_terrier,irish_water_spaniel,irish_wolfhound,italian_greyhound,japanese_spaniel,keeshond,kelpie,kerry_blue_terrier,komondor,kuvasz,labrador_retriever,lakeland_terrier,leonberg,lhasa,malamute,malinois,maltese_dog,mexican_hairless,miniature_pinscher,miniature_poodle,miniature_schnauzer,newfoundland,norfolk_terrier,norwegian_elkhound,norwich_terrier,old_english_sheepdog,otterhound,papillon,pekinese,pembroke,pomeranian,pug,redbone,rhodesian_ridgeback,rottweiler,saint_bernard,saluki,samoyed,schipperke,scotch_terrier,scottish_deerhound,sealyham_terrier,shetland_sheepdog,shih-tzu,siberian_husky,silky_terrier,soft-coated_wheaten_terrier,staffordshire_bullterrier,standard_poodle,standard_schnauzer,sussex_spaniel,tibetan_mastiff,tibetan_terrier,toy_poodle,toy_terrier,vizsla,walker_hound,weimaraner,welsh_springer_spaniel,west_highland_white_terrier,whippet,wire-haired_fox_terrier,yorkshire_terrier"
breeds_list = breeds_list.split(',')

# turns torch tensor into pil image and shows it
def showTorchImage(image):
	image = image.mul(255.0)
	image = transforms.ToPILImage()(image)
	image.show()

class DoggoDataset(Dataset):
	# Dataset of dog photos and their breeds (training examples)

	def __init__(self, csv_file, root_dir):
		# Arguments:
			# csv_file (string): Path to csv file w/ annotations
			# root_dir (string): Directory w/ all images
		self.breeds_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir

	def __len__(self):
		return len(self.breeds_frame)

	# returns a dict {'breed': breed, 'image': image}
	def __getitem__(self, idx):
		# idx: image index (out off 12000-ish)		
		breed = self.breeds_frame.iloc[idx, 1]		
		breed = breeds_list.index(breed)
		img_id = self.breeds_frame.iloc[idx, 0]
		img_name = os.path.join(self.root_dir, 'train/' + img_id + '.jpg')
		image = io.imread(img_name) # image is a numpy array: <width x height x color>


		# transforms every image to be 400x400x3
		image = Image.fromarray(image, 'RGB') # numpy array -> PIL image
		image = transforms.Resize(400)(image)
		image = transforms.RandomCrop(400)(image)

		# PIL image -> (normalized) torch FloatTensor
		image = transforms.ToTensor()(image).float().div(255.0)

		sample = (image, breed)
		return sample

from torch.autograd import Variable
import torch.nn.functional as F

learning_rate = .001
n_epochs = 1
batch_size = 10

train_set = DoggoDataset(csv_file='Doggos/labels.csv', root_dir='Doggos/')
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


class SimpleConv(torch.nn.Module):

	def __init__(self):
		super(SimpleConv, self).__init__()
		self.layer1 = torch.nn.Sequential(
		  	torch.nn.Conv2d(3, 10, 5, 2),
		  	torch.nn.BatchNorm2d(10),
		  	torch.nn.ReLU())
		self.layer2 = torch.nn.Sequential(
			torch.nn.Conv2d(10, 20, 5, 2),
			torch.nn.BatchNorm2d(20),
			torch.nn.ReLU())
		self.layer3 = torch.nn.Sequential(
			torch.nn.Conv2d(20, 40, 5, 2),
			torch.nn.BatchNorm2d(40),
			torch.nn.ReLU()) 
	
		self.pool = torch.nn.MaxPool2d(2, 2)

		self.fc1 = torch.nn.Linear(21160, 120)
		
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.pool(x)
		x = x.view(x.shape[0], -1)
		x = self.fc1(x)
		return x


model = SimpleConv()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

showTorchImage(train_set[234][0]) # image check works

total_step = len(train_loader)
# training network
for epoch in range(n_epochs):
	for i, samples in enumerate(train_loader):
		images, breeds = samples
		outputs = model.forward(images)
		 # Forward pass
		loss = criterion(outputs, breeds)
		print(loss)
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if (i+1) % 100 == 0:
			print(brands[2])
			print(outputs[2])
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, i+1, total_step, loss.item()))


