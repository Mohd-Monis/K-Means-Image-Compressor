import numpy as np
from PIL import Image
import random
import sys

def K_random(X,K):
    indices = random.sample(range(X.shape[0]), K);
    return X[indices];

def get_nearest_centroids(X,centroids):
    m,n = X.shape;
    K = centroids.shape[0];
    idx = np.zeros(m,dtype = np.uint8);

    for i in range(m):
        ci = -1;
        dist = float('inf');
        for k in range(K):
            centroid_dist = 0;
            for j in range(n):
                first = centroids[k][j];
                delta = float(X[i][j]) - centroids[k][j];
                centroid_dist += np.square(delta);
            if (ci == -1 or dist > centroid_dist):
                ci = k;
                dist = centroid_dist;
        idx[i] = ci;
    return idx;


def shift_centroids_to_mean(X, K, idx):
    m,n = X.shape;
    new_centroids = np.zeros((K, n));
    centroid_sum = np.zeros_like(new_centroids);
    centroid_num = np.zeros(K);

    for i in range(m):
        assigned_centroid = int(idx[i]);
        for j in range(n):
            centroid_sum[assigned_centroid][j] += X[i][j];
        centroid_num[assigned_centroid] += 1;
    
    for i in range(K):
        new_centroids[i] = centroid_sum[i] / centroid_num[i];
    
    return new_centroids;

def compute_cost(X, idx, centroids):
    dist = 0;
    m, n = X.shape;
    for i in range(m):
        centroid = int(idx[i]);
        for j in range(n):
            dist += np.square(X[i][j] - centroids[centroid][j]);
    
    return dist / m;
    


def compress_image_mapping(image, K):
    max_iter = 10;
    Cost = -1;
    num_pixels = image.shape[0];
    new_map = np.array((num_pixels,));
    clustered_colors = np.array((K,3));
    k_initial = K_random(image, K);
    centroids = k_initial;
    idx = np.zeros(num_pixels,dtype = np.uint8);
    for i in range(max_iter):
        print("on iteration" , i)
        idx = get_nearest_centroids(image, centroids);
        centroids = shift_centroids_to_mean(image, K, idx);
    
    return centroids, idx;
    
def CompressImage(image_path,k):
    img = Image.open(image_path);
    img = np.array(img);
    l, b, t = img.shape;
    img = img.reshape(l * b, 3);
    clustered_colors, img_mapping = compress_image_mapping(img, k);

    final_compressed = [];

    np.savez_compressed(image_path + '.npz', centroids=clustered_colors, labels=img_mapping, shape=(l, b))
    


def OpenImage(image_path):
    img = Image.open(image_path);
    img = np.array(img);
    l, b, t = img.shape;   
    data = np.load(image_path + '.npz')
    centroids = data['centroids']
    labels = data['labels']
    decluster = centroids[labels].reshape(l, b, 3);  #could produce error;
    image = Image.fromarray(decluster.astype(np.uint8));
    image = image.convert("RGB");
    image.show();



def run_k_means():
    while 1:
        print("Menu")
        print("Select 1 for compressing an image")
        print("Select 2 for opening a compressed image")
        print("Select any other key for exiting")

        option = input("");
        if option == '1':
            path = input("enter the path to the image: ");
            k = int(input('Enter a number from 4 to 8, denoting degree of resolution: '));
            CompressImage(path,k);
            print("image compressed!!");

        elif option == '2':
            path = input("enter the path to original image: ")
            OpenImage(path);
    
        else:
            break;
    

run_k_means();




    