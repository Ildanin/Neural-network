import numpy as np
from random import randint
from math import *

class Network:
    def __init__(self,info,non_linear_func,non_linear_func_derivative,delta_factor=(-1,1),delta_bias=(-1,1),accuracy=2):
        self.info = info
        self.func = non_linear_func
        self.func_derivative = non_linear_func_derivative
        self.accuracy = accuracy
        self.layers = []
        self.layer_results = []
        self.layer_dictionary = []
        for i in range(sum(info[1:])):
            self.layer_dictionary.append(self.layer_definer(i))
        for i in range(1,len(info)):
            self.layers.append(np.append(np.random.randint(delta_factor[0]*10**accuracy,delta_factor[1]*10**accuracy+1,(info[i],info[i-1])),(np.random.randint(delta_bias[0]*10**accuracy,delta_bias[1]*10**accuracy+1,(info[i],1))),axis=1)/(10**accuracy))

    def process(self,data_set):
        self.layer_results = []
        self.layer_results.append(data_set)
        for layer in self.layers:
            self.layer_results.append(np.array(list(map(self.func,layer.dot(np.append(self.layer_results[-1],1))))))
        return(self.layer_results[-1])
    
    def transit(self,network):
        if self != network:
            self.layers = []
            if self.info != network.info:
                self.info = network.info.copy()
                self.layer_dictionary = network.layer_dictionary.copy()
            for weights in network.layers:
                self.layers.append(np.copy(weights))
        
    def layer_definer(self,neuron_number):
        counter = 1
        while neuron_number + 1 - self.info[counter] > 0:
            neuron_number -= self.info[counter]
            counter += 1
        return(counter-1,neuron_number)
    
    def random_mutate(self,mutating_neuron_number,mutations_number,evolution_coefficent,change_range=(-0.5,0.5)):
        for i in range(mutating_neuron_number):
            layer_number, neuron_number = self.layer_dictionary[randint(0,sum(self.info[1:])-1)]
            for j in range(mutations_number):
                delta_factor = evolution_coefficent * randint(int(change_range[0]*10**self.accuracy),int(change_range[1]*10**self.accuracy))/10**self.accuracy
                weight_number = randint(0,self.info[layer_number])
                self.layers[layer_number][neuron_number,weight_number] = round(self.layers[layer_number][neuron_number,weight_number] + delta_factor,self.accuracy)

    def backpropagate(self,answer):
        gradient = []
        cumulative_difference = 2*(self.layer_results[-1]-answer)*np.array(list(map(self.func_derivative,self.layer_results[-1])))
        gradient.insert(0,np.reshape(cumulative_difference,[len(cumulative_difference),1])*np.append(self.layer_results[-2],1))
        for i in range(len(self.info)-3,-1,-1):
            cumulative_difference = self.layers[i+1][:,0:(self.info[i+1])].transpose().dot(cumulative_difference)*np.array(list(map(self.func_derivative,self.layer_results[i+1])))
            gradient.insert(0,np.reshape(cumulative_difference,[len(cumulative_difference),1])*np.append(self.layer_results[i],1))
        return(gradient)
    
    def modify(self,gradient,alpha):
        for i, layer in enumerate(self.layers):
            layer += -alpha*gradient[i]

    def cost(self,answer):
        return(sum((self.layer_results[-1]-answer)**2))

def siqmoid(x):
    if x > 15:
        return(1)
    elif x < -15:
        return(0)
    else:
        return(1/(e**(-x)+1))

def siqmoid_derivative(x):
    if -15 < x < 15:
        return((e**x)/((e**x+1)**2))
    else:
        return(0)

def RELU(x):
    if x < 0:
        return(0)
    else:
        return(x)

def RELU_derivative(x):
    if x < 0:
        return(0)
    else:
        return(1)

def L_RELU(x):
    if x < 0:
        return(0.1*x)
    else:
        return(x)

def L_RELU_derivative(x):
    if x < 0:
        return(0.1)
    else:
        return(1)