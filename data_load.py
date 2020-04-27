import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        #key_pts_copy = 2. * key_pts_copy / image_copy.shape - 1.

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}

class RandomHorizontalFlip(object):
    """Horizontally flip the given image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample: dict with image (numpy array) and keypoints (68x2 array) to be flipped 

        Returns:
            dict with image (numpy array) and keypoints (68x2 array) flipped or not
        """
        if np.random.random() < self.p:
            image, key_pts = sample['image'], sample['keypoints']
            return {'image': np.flip(image, axis=1), 
                    'keypoints': key_pts * (-1, 1) + (image.shape[0], 0)}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

    
def get_transform():
    "Return composed transform to be used in Facial_Keypoints"
    return transforms.Compose([Rescale(250),
                               RandomCrop(224),
                               Normalize(),
                               ToTensor()])


def get_training_dataset():
    "Return FacialKeypointsDataset for training"
    return FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                  root_dir='data/training/',
                                  transform=get_transform())


def get_test_dataset():
    "Return FacialKeypointsDataset for test"
    return FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                  root_dir='data/test/',
                                  transform=get_transform())


def get_data_loaders(batch_size=10, num_workers=0):
    "Return train and test DataLoaders"
    train_loader = DataLoader(get_training_dataset(), 
                              batch_size=batch_size,
                              shuffle=True, 
                              num_workers=num_workers)

    test_loader = DataLoader(get_test_dataset(), 
                             batch_size=batch_size,
                             shuffle=True, 
                             num_workers=num_workers)
    
    return train_loader, test_loader


def evaluate_batch(model, data, useGPU=False):
    "test the model on a batch. Return images, output points, key points"
    # get the input images and their corresponding labels
    images = data['image']
    key_pts = data['keypoints']

    # flatten pts
    key_pts = key_pts.view(key_pts.size(0), -1)

    # convert variables to floats for regression loss
    key_pts = key_pts.type(torch.FloatTensor)
    images = images.type(torch.FloatTensor)

    if useGPU and torch.cuda.is_available():
        images = images.cuda(non_blocking=True)
        key_pts = key_pts.cuda(non_blocking=True)

    # forward pass to get outputs
    output_pts = model(images)

    return images, output_pts, key_pts


def show_all_keypoints(ax, image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    ax.imshow(image, cmap='gray')
    ax.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


def visualize_output(test_images, test_outputs, gt_pts=None, shape=None):
    "visualize the output on a batch of images"
    images = list(test_images)
    if shape is None:
        n = int(np.ceil(np.sqrt(len(images))))
        shape = (n, n)
    fig, subplots = plt.subplots(*shape, figsize=(20,10))
    for i, item in enumerate(images):
        #plt.figure(figsize=(20,10))
        #ax = plt.subplot(1, len(images), i+1)
        ax = subplots.ravel()[i]

        # un-transform the image data
        image = item.data   # get the image from it's wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy().reshape(68, -1)
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        #predicted_key_pts = 0.5 * (predicted_key_pts + 1) * image.shape[:2]
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i].reshape(68, -1)
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(ax, np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        #plt.axis('off')
        ax.axis('off')

    plt.show()


def train_net(model, data_loader, optimizer, criterion, n_epochs, 
              scheduler=None, useGPU=False, n_upd=10):

    # prepare the net for training
    if useGPU and torch.cuda.is_available():
      model.to("cuda:0")
    else:
      model.to("cpu")
    model.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            images, output_pts, key_pts = evaluate_batch(model, data, useGPU)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()
            #scheduler.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            if batch_i % n_upd == n_upd - 1:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/len(data_loader)))
                running_loss = 0.0

    print('Finished Training')

