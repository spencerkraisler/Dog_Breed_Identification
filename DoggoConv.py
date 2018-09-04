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

# returns one hot vector of dog breeds
def getOneHotVector(breed):
	idx = breeds_list.index(breed)
	one_hot_vector = torch.zeros(120)
	one_hot_vector[idx] = 1
	return one_hot_vector

# turns a list of dog breed indecies to a list of one hot vectors (for dataloader)
def indecies_to_one_hot_vectors(indecies):
	one_hot_vectors = torch.zeros((indecies.shape[0], 120))
	for i in range(indecies.shape[0]):
		idx = indecies[i]
		one_hot_vector = torch.zeros(120)
		one_hot_vector[idx] = 1
		one_hot_vectors[i] = one_hot_vector
	return one_hot_vectors

# turns torch tensor into pil image and shows it
def showTorchImage(image):
	image = image.div(255.0)
	image = transforms.ToPILImage()(image)
	image.show()

class DoggoDataset(Dataset):
	# Dataset of dog photos and their breeds (training examples)

	def __init__(self, csv_file, root_dir, transform=None):
		# Arguments:
			# csv_file (string): Path to csv file w/ annotations
			# root_dir (string): Directory w/ all images
			# transform (callable, optional): Option transform 
			# to be applied on a sample
		self.breeds_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.breeds_frame)


	# returns a dict {'breed': breed, 'image': image}
	def __getitem__(self, idx):
		# idx: image index (out off 12000-ish)		
		breed = self.breeds_frame.iloc[idx, 1]		
		breed = getOneHotVector(breed)
		img_id = self.breeds_frame.iloc[idx, 0]
		img_name = os.path.join(self.root_dir, 'train/' + img_id + '.jpg')
		image = io.imread(img_name) # image is a numpy array: <width x height x color>
		image = Image.fromarray(image, 'RGB')
		sample = {'breed': breed, 'image': image}

		if self.transform: 
			sample = self.transform(sample)

		return sample

class Resize(object):
	# Rescale the image in a sample to a given size

	def __init__(self, output_size):
		# Arguments:
			# output_size (tuple or int): Desired output size. 
			# If tuple, output is matched to output_size. If 
			# int, smaller of image edges is matched to 
			# output_size keeping aspect ratio the same.
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		breed, image = sample['breed'], sample['image']
		image = transforms.Resize(self.output_size)(image)
		return {'breed': breed, 'image': image}

class RandomCrop(object):
	# Crop randomly the image in a sample

	def __init__(self, output_size):
		# Argument:
			# output_size (tuple or int): Desired 
			# output size. If int, square crop is made.
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		breed, image = sample['breed'], sample['image']
		h, w = image.size
		new_h, new_w = self.output_size
		image = transforms.RandomCrop([new_h, new_w])(image)
		return {'breed': breed, 'image': image}
class ToNormalizedTensor(object):
	# Convert ndarrays in sample to torch Tensors

	def __call__(self, sample):
		breed, image = sample['breed'], sample['image']
		image = np.array(image)
		# swap color axis because
		# numpy image: <height x width x color>
		# torch image: <color x height x width>
		image = image.transpose((2,0,1))
		image = torch.from_numpy(image).float()
		return {'breed': breed, 'image': image}

from torch.autograd import Variable
import torch.nn.functional as F

learning_rate = .001
n_epochs = 6
batch_size = 10

train_set = DoggoDataset(csv_file='Doggos/labels.csv', root_dir='Doggos/', transform=transforms.Compose([Resize(300), 
																									   RandomCrop(300), 
																									   ToNormalizedTensor()]))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

class SimpleConv(torch.nn.Module):

	def __init__(self):
		super(SimpleConv, self).__init__()
	
		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv2d(3, 10, kernel_size=5, stride=2, padding=2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=4, stride=2))

		self.layer2 = torch.nn.Sequential(
			torch.nn.Conv2d(10,20, kernel_size=5, stride=2, padding=2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2))

		self.out = torch.nn.Linear(6480, 120)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.view(x.size(0), -1)
		x = self.out(x)
		return x

model = SimpleConv()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

# training network
for epoch in range(n_epochs):
	for i, samples in enumerate(train_loader):
		images = samples['image']
		breeds = samples['breed']
		outputs = model.forward(images)
		breeds = breeds.view(breeds.size(0),-1)
		outputs = outputs.view(outputs.size(0),-1)
		 # Forward pass
		loss = criterion(outputs, breeds)
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if (i+1) % 100 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, i+1, total_step, loss.item()))


correct = 0
total = 0

# evaluation
for i, samples in enumerate(train_loader):
	images = samples['image']
	breeds = samples['breed']
	outputs = model.forward(images)
	predicts = torch.max(outputs, 1)
	predicts = predicts[1]
	predicts = indecies_to_one_hot_vectors(predicts)
	product = predicts * breeds
	correct += int(product.sum())
	total += breeds.size(0)
	if i % 100 == 0:
		print(i, correct, total)




