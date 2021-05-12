import random
import math
import pandas as pd
from prettytable import PrettyTable
from tabulate import tabulate


class Network:
	def __init__(self, structure, learning_rate, bias):
		self.layers, self.layer_names = self.getNetwork(structure)
		self.learning_rate = learning_rate
		self.bias = bias

	def getNetwork(self,structure):
		layers = dict()

		layer = dict()
		layer["nodes"] = [0 for i in range(0,structure[0])]

		layer["weights"] = []
		layer["deltas"] =  [0 for i in range(0,structure[0])]

		layers["input"] = layer
		layer_names = []
		layer_names.append("input")

		prev_nodes = structure[0]
		current_nodes = 0

		for layer_number, struct_nodes in enumerate(structure[1:-1]):

			current_nodes = struct_nodes
			layer = dict()
			layer["nodes"] = [0 for i in range(0,struct_nodes)]

			layer["weights"] = []
			layer["deltas"] = [0 for i in range(0,struct_nodes)]
			layer_name = "hidden_" + str(layer_number+1)
			layer_names.append(layer_name)
			layers[layer_name] = layer
			layers[layer_name]["weights"] = self.getWeightMatrix(layer_number, current_nodes, prev_nodes)
			prev_nodes = current_nodes



		layer = dict()
		layer["nodes"] =  [0 for i in range(0, structure[-1])]

		layer["weights"] = self.getWeightMatrix(len(structure)-2, structure[-1], prev_nodes)
		layer["deltas"] = [0 for i in range(0, structure[-1])]

		layers["output"] = layer
		layer_names.append("output")

		return layers,layer_names






	def train(self,X_values,Y_values,epochs):
		for epoch in range(0,epochs):
			correct = 0
			print("Epoch :" ,epoch)
			for X ,Y in zip(X_values,Y_values):

				O = self.forwardPropogate(X)

				# print(O,Y)

				if O.index(max(O)) == Y.index(max(Y)):
					correct += 1
				
				self.backPropogate(Y)
				# self.displayNetwork()
				self.updateWeights()
				# self.displayNetwork()
			print("Error is ", self.LMS(O,Y))
			print("Accuracy : ",correct/len(X_values))


	def test(self,X_values,Y_values):

		table = PrettyTable(['Predicted Value','Golden Truth'])
		count = 0
		for X, Y in zip(X_values,Y_values):

			O = self.forwardPropogate(X)

			predicted = categories_dict[O.index(max(O))]
			actual = categories_dict[Y.index(max(Y))]
			if predicted == actual:
				count = count + 1
			table.add_row([predicted,actual])

		print("Prediction Results")
		print(table)
		print("\n The Accuracy is: ",count/len(X_values))




	def updateWeights(self):
		for index, current_layer in enumerate(self.layer_names[:-1]):

			next_layer = self.layer_names[index+1]

			for  j, delta_j in enumerate(self.layers[next_layer]["deltas"]):

				for i, X_i in enumerate(self.layers[current_layer]["nodes"]):

					delta_w = self.learning_rate*delta_j*X_i

					if self.layers[next_layer]["weights"][j][i] != "ZERO":
						self.layers[next_layer]["weights"][j][i] += delta_w


			




	def forwardPropogate(self,input_values):
		self.layers["input"]["nodes"] = input_values
		prev_layer_X = self.layers["input"]["nodes"]
		for current_layer in self.layer_names[1:]:

			for i in range(0,len(self.layers[current_layer]["nodes"])):

				summation_X = self.dotProduct(self.layers[current_layer]["weights"][i], prev_layer_X)
				
				self.layers[current_layer]["nodes"][i] = self.sigmoid(summation_X)

			prev_layer_X = self.layers[current_layer]["nodes"]

		return self.layers["output"]["nodes"]



	def backPropogate(self,target_values):
		T = target_values

		for k in range(0,len(self.layers["output"]["nodes"])):
			O_k = self.layers["output"]["nodes"][k]
			T_k = T[k]
			self.layers["output"]["deltas"][k] =  self.derivative_sigmoid(O_k)*(T_k - O_k)


		next_layer = "output"

		for current_layer in reversed(self.layer_names[:-1]):

			for h in range(0,len(self.layers[current_layer]["nodes"])):
				O_h = self.layers[current_layer]["nodes"][h]
				self.layers[current_layer]["deltas"][h] =  self.derivative_sigmoid(O_h)*self.getWeightsDeltaProduct(current_layer,next_layer,h)
			next_layer = current_layer




	def getWeightsDeltaProduct(self,current_layer,next_layer,node):

		#Edges generating out from current Node
		edges = len(self.layers[next_layer]["nodes"])
		deltas = self.layers[next_layer]["deltas"]
		W = self.layers[next_layer]["weights"]
		h = node


		return sum( 0 if str(W[k][h]) == "ZERO" else W[k][h]*deltas[k] for k in range(0,edges) )






	def LMS(self,Y1,Y2):

		return sum( (y1 - y2)**2 for y1,y2 in zip(Y1, Y2))/2


	def derivative_sigmoid(self,Y):
		return Y*(1 - Y)


	def sigmoid(self,x):
		return 1 / (1 + math.exp(-x))

	def getWeightMatrix(self,layer_number,rows,cols):


		links_for_layer = links[layer_number]
		return [[random.uniform(-0.5,0.5) if links_for_layer[i][j] == "1" else "ZERO" for j in range(0,cols)] for i in range(0,rows) ]



	def dotProduct(self,weights,inputs):
		if len(weights) != len(inputs):
			print("Error dimensions should be same")
			exit(0)
		return sum(0 if str(i[0]) == "ZERO" else i[0]*i[1] for i in zip(weights, inputs)) + self.bias




	def displayNetwork(self):
		for layer_name,value in self.layers.items():
			print("\nLAYER :"+layer_name.upper())
			print("Nodes\n",tabulate([value["nodes"]]))
			print("Weights\n",tabulate(value["weights"]))
			print("Deltas\n",tabulate([value["deltas"]]))


#Number of nodes in Each Layer
structure = [4,4,3,3]

#Links represented as a Binary String for every node
links = [["1110","1111","1111","1001"],["0011","1110","0011"],["011","011","111"]]

if __name__ == '__main__':



	#Stochaistic Gradient Descent
	network = Network(structure, learning_rate = 0.5, bias = 0.5)
	print("Initial Network")
	network.displayNetwork()


	#Weight Matrix corresponding to a layer denotes weights from previous layer to current layer node.
	#Hence input layer has no weight matrix


	#Loading IRIS Dataset Constructing train_X and train_Y

	train_X = []
	train_Y = []
	df = pd.read_csv('iris.csv')
	df.head(10)

	feature_matrix = df.iloc[:, [0,1,2,3]].values
	train_X = feature_matrix

	gold_truth_labels = df.iloc[:, 4].values

	for label in gold_truth_labels:
		if label == "setosa":
			train_Y.append([1,0,0])
		elif label == "versicolor":
			train_Y.append([0,1,0])
		else:
			train_Y.append([0,0,1])

	#Creating Global Variable Categories Dictionary to match Output Label with Output Vector
	global categories_dict
	categories_dict = dict()
	categories_dict[0] = "setosa"
	categories_dict[1] = "versicolor"
	categories_dict[2] = "virginica" 


	#Training the Model
	network.train(train_X,train_Y,20)
	print("Network After Training")
	network.displayNetwork()



	#Testing the Model
	test_X = [ [6.4, 2.9, 4.3, 1.3], [6.6, 3, 6.4, 1.4], [6.8, 2.8, 8, 1.4], [6.7, 3, 1, 1.7]]

	test_Y = [[0,0,1],[0,0,1],[0,1,0],[1,0,0]]


	network.test(test_X,test_Y)