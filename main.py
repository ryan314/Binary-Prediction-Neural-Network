import numpy as np
import datetime 
import os.path
import sys


# HELPER FUNCTIONS
def wr_predic(name, cat1, cat2, out):
  if (abs(1 - out) <= abs(0 - out)):
    file2.write(name + ": %.3f" % out + " (" + cat1 + ")\n")
  else:
    file2.write(name + ": %.3f" % out + " (" + cat2 + ")\n")

def sigmoid(x):
  # Activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid activation function: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are both arrays of the same length
  return ((y_true - y_pred) ** 2).mean()


# MEMORY FEATURE SETUP
if os.path.exists(f"MEMORY") == False:
  os.mkdir(f"MEMORY")
  os.chdir(f"MEMORY")
else:
  os.chdir(f"MEMORY")

if os.path.exists(f"NN_Mem.txt") == False:
  file1 = open(f"NN_Mem.txt", "w")
  file1.write("0.1" + "\n")
  file1.write("1000" + "\n")
  os.chdir("..") # goes up 1 directory
else:
  file1 = open(f"NN_Mem.txt", "r")
  os.chdir("..") 

str1 = file1.readline()
str2 = file1.readline()
strlen1 = len(str1)
strlen2 = len(str2)

learn_rate = float(str1[0:strlen1-1])
epochs = int(str2[0:strlen2-1])
file1.close() 


# EDIT MEMORY FEATURE 
edit_bool = input("Would you like to edit the MEMORY of the neural network? (y/n)\n")
print("\n")

if edit_bool == "y":
  file1 = open(f"MEMORY/NN_Mem.txt", "w")
  upd_learn_rate = input("What would you like to update learn_rate to?\n")
  print("\n")
  file1.write(upd_learn_rate + "\n")
  upd_epochs = input("What would you like to update epochs to?\n")
  print("\n")
  file1.write(upd_epochs + "\n")
  file1.close()


# LOGGING FEATURE SETUP 
if os.path.exists(f"LOGS") == False:
  os.mkdir(f"LOGS")
  os.chdir(f"LOGS")
else:
  os.chdir(f"LOGS")

curr_moment = datetime.datetime.now()
file_name = curr_moment.strftime("%Y-%m-%d_%H.%M.%S") + "_Log"
file2 = open(file_name + ".txt", "w")
file2.write("------- EPOCH RESULTS -------\n\n")
os.chdir("..")

class NeuralNetwork:
  def __init__(self):
    # Randomly Initializing Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Randomly Initializing Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a 2-element numpy array
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
        - Note: n = # of samples within dataset
        - data is an n-element numpy array (where each element is a 2-element list)
        - all_y_trues is an n-element numpy array corresponding to each element within data 
    '''
  
    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # Feedforward Process
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # Partial Derivative Calculations
        # Naming Convention: d_L_d_w1 means "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Update Weights and Biases with Stochastic Gradient Descent (SGD)
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # Calculate Loss at End of Each 10 Epochs
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        file2.write("Epoch %d, Loss: %.3f" % (epoch, loss))
        file2.write("\n")

# DEFINE DATASET 
int_n1 = -1
int_n2 = -1
idx1 = 0
idx2 = 0
int_cat = -1
loop_amt = 0
temp = "-1"

data = []
all_y_trues = []

cat1 = input("Category 1 (will be assigned binary value of 1): ")
print("\n")
cat2 = input("Category 2 (will be assigned binary value of 0): ")
print("\n")

loop_amt = int(input("How many data entries would you like to enter?\n"))
print("\n")

while (loop_amt > 0):
  temp = input("N1: ")
  int_n1 = int(temp)

  temp = input("N2: ")
  int_n2 = int(temp)

  data.insert(idx1, [int_n1, int_n2])
  idx1 += 1

  temp = input("Category (int): ")
  int_cat = int(temp)
  all_y_trues.insert(idx2, int_cat)
  idx2 += 1

  print("\n")
  loop_amt -= 1


# TRAIN THE NEURAL NETWORK 
network = NeuralNetwork()
network.train(data, all_y_trues)

input1 = input("Input 1: ")
int_n1 = int(input("N1: "))
int_n2 = int(input("N2: "))
arr1 = [int_n1, int_n2]
print("\n")

input2 = input("Input 2: ")
int_n1 = int(input("N1: "))
int_n2 = int(input("N2: "))
arr2 = [int_n1, int_n2]
print("\n")

out1 = network.feedforward(arr1)
out2 = network.feedforward(arr2)

# WRITE PREDICTIONS
file2.write("\n\n")
file2.write("------- PREDICTIONS -------\n\n")
wr_predic(input1, cat1, cat2, out1)
wr_predic(input2, cat1, cat2, out2)
file2.close()

print("The log has been successfully generated!")
exit = input("Press ENTER to exit.")