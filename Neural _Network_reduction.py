import pandas as pd
import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import math
import warnings
warnings.filterwarnings('ignore')



# reading the SFEW dataset
print("Reading data")
print("........................")
data=pd.read_excel('SFEW.xlsx')
data.rename(columns = {'Unnamed: 0':'Name', 'label':'Label',
       'First 5 Principal Components of Local Phase Quantization (LPQ) features\nLocal Phase Quantization (LPQ) features':'I1',
       'Unnamed: 3':'I2', 'Unnamed: 4':'I3', 'Unnamed: 5':'I4', 'Unnamed: 6':'I5',
       'First 5 Principal Components of Pyramid of Histogram of Gradients (PHOG) features':'I6',
       'Unnamed: 8':'I7', 'Unnamed: 9':'I8', 'Unnamed: 10':'I9', 'Unnamed: 11':'I10'}, inplace = True)

print("Pre-Processing data")
print("........................")
working_data= data.drop(['Name'],axis=1)
working_data = working_data.fillna(working_data.mean())
working_data['Label']=working_data['Label'] - 1
working_data = working_data.sample(frac=1).reset_index(drop=True)

#Spliiting into inut and target columns
X = working_data.iloc[:,1:]
y = working_data.iloc[:,0]

#Splitting traiing data into training and testing set (366 samples as training) rest as testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.487, random_state=42)

#Fitting min-max scaler into our training inputs(0-1 Range)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

#creating tensor of training and testing data
X_train = torch.tensor(X_train  ,dtype = torch.float)
y_train = torch.tensor(y_train  ,dtype = torch.long)
X_test = torch.tensor(X_test  ,dtype = torch.float)
y_test = torch.tensor(y_test  ,dtype = torch.long)

# initialising our input and output size and  hyperparameters for our model 
input_neurons = 10
hidden_neurons = 20
output_neurons = 7
learning_rate = 0.04
num_epoch = 500

#creating a class for our basic 3 layer network
print("Building Model")
print(".......................")
class Net(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Net,self).__init__()
        self.layer1 = torch.nn.Linear(n_input,n_hidden)
        self.outer_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self,x):
        h_input = self.layer1(x)
        h_output = F.relu(h_input)
        y_pred = self.outer_layer(h_output)

        return y_pred,h_output

net = Net(input_neurons, hidden_neurons, output_neurons)
# defing which loss to use
loss_func = torch.nn.CrossEntropyLoss()
#defing optimiser
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

# creating loss list for visualizing
all_losses = []

#training ourr network
print('""""Model Training Started"""" \n """""""""""""""""')
for epoch in range(num_epoch):
    #performing forward pass 
    Y_pred,h_final = net(X_train)
    #calculating loss
    loss = loss_func(Y_pred,y_train)
    all_losses.append(loss.item())
    #print progress at every 50 epochs
    if epoch % 50 == 0:
        _, predicted = torch.max(F.softmax(Y_pred,1), 1)

        total = predicted.size(0)
        correct = predicted.data.numpy() == y_train.data.numpy()
        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epoch, loss.item(), 100 * sum(correct)/total))
    #clearing the gradients
    net.zero_grad()
    #performing backward pass
    loss.backward()
    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

print("Model Training done\n ...................")
#visualising loss
fig,ax = plt.subplots()
ax.plot(all_losses)

#Accuracy of our model in the test set
print("Testing our basic model\n ................")
Y_pred_test,h_test = net(X_test)
_, predicted_test = torch.max(F.softmax(Y_pred_test,1), 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == y_test.data.numpy())

print('Testing Accuracy of basic Model: %.2f %%' % (100 * correct_test / total_test))


#class wise precision recall and f1-score
print("Evaluation of our  basic model on test set")
testing_pred = predicted_test.data.numpy()
actual = y_test.data.numpy()
target_names = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
print(classification_report(testing_pred,actual,target_names=target_names))


print("Computing Angles between hidden pattern vectors\n ..........")
#scaling our pattern vectors into the range -0.5 to 0.5 (normal min max then subtracting 0.5 ) 
k_array=np.array(h_final.detach())
pattern_vector=k_array.T
pattern_vector=scaler.fit_transform(pattern_vector)
pattern_vector=pattern_vector -0.5
pattern_vector = pattern_vector.tolist()

#finding pairs (combinations) of the pattern vector
def find_tuples_index(lst,num=2):
     return [i for i in itertools.combinations(enumerate(lst), num)]
pair_angle=find_tuples_index(pattern_vector,num=2)

#Function to compute angle

def find_angle(vector1,vector2): 
    vector_1=torch.tensor(vector1 , dtype = torch.float)
    vector_2=torch.tensor(vector2 , dtype = torch.float)
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return math.degrees(angle)

angles = []
indices = []
main =[]
for i in range(len(pair_angle)):
    #finding angles between the pairs in pair_angle
    angles.append(find_angle(pair_angle[i][0][1],pair_angle[i][1][1]))
    #indices of the angle 
    indices.append((pair_angle[i][0][0],pair_angle[i][1][0]))
for j in range(len(angles)):
    main.append((angles[j] , indices[j]))
main=np.array(main)

#creating list for appending all similar pairs index
similar=[]
for k in range(main.shape[0]):
    if main[k][0] <= 15:
        similar.append((main[k][0] ,main[k][1]))
#creating list for appending all cpmplementary pairs index
complimentary=[]
for l in range(main.shape[0]):
    if main[l][0] >= 165:
        complimentary.append((main[l][0] ,main[l][1]))


#cloning the weights for calculation
weights_new = net.layer1.weight.data
similar_weights = weights_new.clone()
out_weights = net.outer_layer.weight.data
old_out_weights = out_weights.clone()
bias  = net.layer1.bias.data
old_bias = bias.clone()
print("Removing complementary neurons and adding similar weights\n .......")
# creating a list for the index of neurons that are complementary and needs to be removed
to_remove_comp = []
for i in complimentary:
    if i[1][0] not in to_remove_comp:
        to_remove_comp.append(i[1][0])
    if i[1][1] not in to_remove_comp:
        to_remove_comp.append(i[1][1])
#Changing the rows of weight matrix and bais for the corresponding neurons to zero and the column of output weights to zero
for i in to_remove_comp:
    weights_new[i]=0.0
    out_weights[:,i]=0.0
    bias[i]=0.0

#Adding weights of the similar indices and then removing one of them
for i in similar:
    weights_new[i[1][1]]+=similar_weights[i[1][0]]
    out_weights[:,i[1][1]]+=old_out_weights[:,i[1][0]]
    bias[i[1][1]]+=old_bias[i[1][0]]
#After adding removing the neurons for which we have added the weight to another 
for i in similar:
    weights_new[i[1][0]] = 0
    out_weights[:,i[1][0]]=0
    bias[i[1][0]]=0
c=0
for i in range(weights_new.shape[0]):
    if weights_new[i].detach().numpy().all()== 0:
        c+=1
print("Number of Neurons Pruned (out of 20)",c)
# Creating a new model and initilaising the input hidden and output layers with the old class
# Editing the weight matrix , output weight and bias and replacing them with our new reduced weights and bias.
print("Building Reduced Model")
class Reduced_Net(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Reduced_Net,self).__init__()
        self.layer_1 = net.layer1
        self.layer_1.weight.data = weights_new
        self.layer_1.bias.data = bias
        self.outer_layer = net.outer_layer
        self.outer_layer.weight.data = out_weights

    def forward(self,x):
        h_input = self.layer_1(x)
        h_output = F.relu(h_input)
        
        y_pred = self.outer_layer(h_output)

        return y_pred


net1 = Reduced_Net(input_neurons, hidden_neurons, output_neurons)
print(" '''''New Reduced Model created ''''' ")

# testing our new reduced model on the testing set
print("Testing reduced model on testing data")
Y_pred_test= net1(X_test)
_, predicted_test_r = torch.max(F.softmax(Y_pred_test,1), 1)
total_test = predicted_test_r.size(0)
correct_test = sum(predicted_test_r.data.numpy() == y_test.data.numpy())

print('Testing Accuracy of reduced Model : %.2f %%' % (100 * correct_test / total_test))


#class wise precision recall and f1-score for our new reduced model
print("Evaluation measures for new reduced model ")
testing_pred = predicted_test_r.data.numpy()
actual = y_test.data.numpy()
target_names = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
print(classification_report(testing_pred,actual,target_names=target_names))



print("Accuracies of model before and after Pruning(ran 5 times) and checked")
print(".................................")

neurons_pruned =[9,10,8,11,6]
Testing_accuracy_after_reducing = [19.93,20.97,14.69,19.76,17.93]
normal_testing = [17.9,20.06,17.63,19.45,18.84]

ind = np.arange(5)# the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)

rects1 = ax.bar(ind+width, Testing_accuracy_after_reducing, width, color='r',align='center')
rects2 = ax.bar(ind,normal_testing, width, color='b',align = 'center')
ax.set_xlabel(" No. of Hidden Units Pruned",fontsize = 16)
ax.set_ylabel(" Testing Accuracy",fontsize=16)
ax.set_xticks(ind+width)
ax.set_xticklabels(('9','10','8','11','6'))

ax.legend(['After Pruning','Before Pruning'],loc=1)

plt.show()




