import os
import random
import pickle
import math

import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


out_dir = './dataset'

class PointCloudData(Dataset):
    def __init__(self, data):

        #save data
        self.data = data

        #save different classes
        tempClasses = pd.Series(data['category']).unique()
        self.classes = {myClass: i for i, myClass in enumerate(tempClasses)}

        #change data
        self.data.category = [self.classes[item] for item in self.data.category]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pointcloud = self.data['pointcloud'][idx]
        category = self.data['category'][idx]

        return {'pointcloud': pointcloud,'category': category}

def createData():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if len(os.listdir(out_dir)) == 0:
        #todo download data automatically
        pass

def readOff(path):

    with open(path) as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')

        n_vertices, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        vertices = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_vertices)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]

    return [vertices, faces]

def collectData(path, k=3000):

    outputTrain = []
    outputTest = []

    #iterate through all files in folder and get type, category and file
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if file.endswith(".off"):
                print(file)
                temp = root.split("\\")
                type = temp[-1]
                category = temp[-2]
                data = readOff(root + "/" + file)

                data = samplePoints(data, k=k)
                data = augmentPoints(data)

                if type == "test":
                    outputTest.append([category, data])
                if type == "train":
                    outputTrain.append([category, data])

    with open(path + "/testData.pickle", "wb") as file:
        pickle.dump(outputTest, file)
    with open(path + "/trainData.pickle", "wb") as file:
        pickle.dump(outputTrain, file)

def readData(path):

    with open(path, 'rb') as file:
        data = pd.DataFrame(pd.read_pickle(file))

    data.columns = ["category", "pointcloud"]

    def clean(x):
        x = x[0]
        x = torch.from_numpy(x)
        return(x)

    data['pointcloud'] = data['pointcloud'].apply(clean)

    return data

def visualizeModel(data):
    vertices = data[0]
    x,y,z = np.array(vertices).T

    faces = data[1]
    i, j, k = np.array(faces).T

    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)])
    fig.show()

def visualizePoints(data):

    vertices = data[0]

    x,y,z = np.array(vertices).T

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
    fig.show()

def samplePoints(data, k=3000):

    #extract existing information
    faces = data[1]
    vertices = np.array(data[0])

    #create empty array that will store the size of a face
    areas = np.zeros((len(faces)))

    #calculate size of face
    def getSize(pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    #iterate all faces
    for i in range(len(areas)):

        #calculate and store size of face
        areas[i] = getSize(vertices[faces[i][0]],
                              vertices[faces[i][1]],
                              vertices[faces[i][2]])

    #select randomly k faces based on their weights
    sampled_faces = (random.choices(faces,
                        weights=areas,
                        k=k))

    # function to sample points on a triangle surface
    def sample_point(pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s) * pt2[i] + (1-t) * pt3[i]
        return (f(0), f(1), f(2))

    #this np array will contain the points
    pointcloud = np.zeros((k, 3))

    # sample points on chosen faces for the point cloud of size 'k'
    for i in range(len(sampled_faces)):
        pointcloud[i] = (sample_point(vertices[sampled_faces[i][0]],
                                      vertices[sampled_faces[i][1]],
                                      vertices[sampled_faces[i][2]]))

    #store in a list for the visualization
    return [pointcloud]

def augmentPoints(data):
    data = data[0]

    #normalize
    norm_data = data - np.mean(data, axis=0)
    norm_data /= np.max(np.linalg.norm(data, axis=1))

    # rotation around z-axis
    theta = random.random() * 2. * math.pi # rotation angle
    rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                           [ math.sin(theta),  math.cos(theta),    0],
                           [0,                             0,      1]])

    rot_data = rot_matrix.dot(norm_data.T).T

    # add some noise
    noise = np.random.normal(0, 0.02, (data.shape))
    noisy_data = rot_data + noise

    return [noisy_data]
