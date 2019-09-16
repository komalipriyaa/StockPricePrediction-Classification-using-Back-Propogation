#STOCK PRICE PREDICTION- Classification Problem

#imports
from sklearn.model_selection import train_test_split     
import numpy as np

#Reading csv file using numpy
wines = np.genfromtxt("/home/komali_priya/Documents/google.csv", delimiter=",", skip_header=0)
print("wines  ",wines)

#Normalizing
wines = wines.T
maxArray = []
for i in range (1 , len(wines)):
    maximum = max(wines[i])
    maxArray.append(maximum)
    k = 0
    for j in wines[i]:
        
        j = j/maximum
        wines[i][k] = j
        k = k+1
 
    
print(maxArray)
wines = wines.T

X = []
for i in range(len(wines)-1):
    X.append(wines[i])
X=np.asarray(X)

print("X is  ",X)

# Random Splitting of training and testing data 
X_train, X_test = train_test_split(X, test_size=0.1, random_state=27)
print("X_train is  ",X_train)

#Creating target lists
y_train = []
y_test = []
for i in X_train:
    j = int(i[0])
    j = j+1
    y_train.append(wines[j][1])
    
for i in X_test:
    j = int(i[0])
    j = j+1
    y_test.append(wines[j][1])

print("y_train",y_train)
train_target = []
for i in range(len(X_train)) :
    if (X_train[i][1] < y_train[i] ):
        train_target.append(1)     #"1" representing increase in stock price
    else:
         train_target.append(0)   #"0" representing decrease in stock price
print("train_target  ",train_target)

test_target = []
for i in range(len(X_test)) :
    if (X_test[i][1] < y_test[i] ):
        test_target.append(1)
    else:
         test_target.append(0)
print("test_target  ",test_target)

x0 = 1
flag = 0

w50 = 0.2
w51 = 0.2
w52 = 0.2
w53 = 0.2
w54 = 0.2

w60 = 0.2
w61 = 0.2
w62 = 0.2
w63 = 0.2
w64 = 0.2

w70 = 0.2
w75 = 0.2
w76 = 0.2

ww50 = 0
ww51 = 0 
ww52 = 0 
ww53 = 0 
ww54 = 0 

ww60 = 0 
ww61 = 0 
ww62 = 0 
ww63 = 0
ww64 = 0 

ww70 = 0 
ww75 = 0  
ww76 = 0

def function (x1,x2,x3,x4,w50,w51,w52,w53,w54,w60,w61,w62,w63,w64,w70,w75,w76,flag,target,ww50,ww51,ww52,ww53,ww54,ww60,ww61,ww62,ww63,ww64,ww70,ww75,ww76):
    flag = flag + 1
    n = 0.1
    
    h1 = x0*w50 + x1*w51 + x2*w52 + x3*w53 + x4*w54
    h2 = x0*w60 + x1*w61 + x2*w62 + x3*w63 + x4*w64 
    
    output = x0*w70 + h1*w75 + h2*w76
    
    dout = output*(1-output)*(target-output)
    
    dhidden1 = h1*(1-h1)*(w75*dout)
    dhidden2 = h2*(1-h2)*(w76*dout)
    
    ww75 = n*dout*h1
    ww76 = n*dout*h2
    ww70 = n*dout*x0
    
    ww50 = n*dhidden1*x0
    ww51 = n*dhidden1*x1
    ww52 = n*dhidden1*x2
    ww53 = n*dhidden1*x3
    ww54 = n*dhidden1*x4
    
    ww60 = n*dhidden2*x0
    ww61 = n*dhidden2*x1
    ww62 = n*dhidden2*x2
    ww63 = n*dhidden2*x3
    ww64 = n*dhidden2*x4
    
    w50 = w50 + ww50
    w51 = w51 + ww51
    w52 = w52 + ww52
    w53 = w53 + ww53
    w54 = w54 + ww54
    
    w60 = w60 + ww60
    w61 = w61 + ww61
    w62 = w62 + ww62
    w63 = w63 + ww63
    w64 = w64 + ww64
    
    return w50,w51,w52,w53,w54,w60,w61,w62,w63,w64,w70,w75,w76,ww50,ww51,ww52,ww53,ww54,ww60,ww61,ww62,ww63,ww64,ww70,ww75,ww76

   
# Iterations
for i in range(35000):
    print("Iteration = ",i)
    for k in range(len(X_train)):
        x1 = X_train[k][1]
        x2 = X_train[k][2]
        x3 = X_train[k][3]
        x4 = X_train[k][4]
        target = train_target[k]
        w50,w51,w52,w53,w54,w60,w61,w62,w63,w64,w70,w75,w76,ww50,ww51,ww52,ww53,ww54,ww60,ww61,ww62,ww63,ww64,ww70,ww75,ww76 = function (x1,x2,x3,x4,w50,w51,w52,w53,w54,w60,w61,w62,w63,w64,w70,w75,w76,flag,target,ww50,ww51,ww52,ww53,ww54,ww60,ww61,ww62,ww63,ww64,ww70,ww75,ww76)
#TESTING
        
output01 = []  
for k in range(len(X_test)):
    x1 = X_test[k][1]
    x2 = X_test[k][2]
    x3 = X_test[k][3]
    x4 = X_test[k][4]
    target = test_target[k]
    
    h1 = x0*w50 + x1*w51 + x2*w52 + x3*w53 + x4*w54
    h2 = x0*w60 + x1*w61 + x2*w62 + x3*w63 + x4*w64 
    
    output = x0*w70 + h1*w75 + h2*w76
    loss = ((target-output)**2)/2
    
    print("Loss : ",loss)
    print("output : ",output)   
    print("target : ",target)
    print("\n")
    if( output<0.5 ):
        output01.append(0)
    else:
        output01.append(1)
print("Target List :\n ",test_target)
print("Output List :\n ",output01) 

#Accuracy calculation
accuracy = 0
total = len(test_target) 
print("No. of testing tuples: ",total)
for i in range(len(test_target)):
    if( test_target[i] == output01[i] ):
        accuracy = accuracy+1
percentage = (accuracy*100)/total

print("Accuracy :  ",percentage)
        
   
    




