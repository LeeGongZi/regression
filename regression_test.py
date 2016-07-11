import numpy as np

##---------------------------------prepare data---------------------------------------#
dataFile = open("data.csv","r")
data = dataFile.readlines()


#delete the first line of data
del data[0]

X = []
Y = []

for i in data:
    #y = w0+w1*x1+w2*x2+w3*x3,
    #or it can be writed in this format:y = w0*x0+w1*x1+w2*x2+w3*x3
    #here, x0 = 1 always. so the next line code just add an x0 = 1 in each data
    x = [1.]
    t = i.split(',')
    #the last is y, so it just be distributed the first three data
    for i in t[:-1]:
        x.append(float(i))
    X.append(x)
    #the last y
    Y.append(float(t[-1]))

dataFile.close()

########################################################################################
#--------------First Method:     Using Matrix to solve the problem---------------------#

#initial the two matrix X,Y with data above
X_matrix = np.matrix(X)
Y_matrix = np.matrix(Y).T

#just a simple matrix calculation to obtain the w
w = (X_matrix.T*X_matrix).I*X_matrix.T*Y_matrix
print w

#-----------------------------------------------------------------------------------------


########################################################################################
#--------------Second Method:Gradient Decent-------------------------------------------#
#this method is an special sample of the third method when ajust the batch_size--------#


learningRate = 0.001
epoch = 1000
#initialize the w0,w1,w2,w3
w = [0,1,1,1]

for _ in range(epoch):
    for i in range(len(X)):
    #Gradient Decent One By One
        y = 0
        for j in range(4):
            y = y + w[j]*X[i][j]
        delta_w =[]
        for xi in X[i]:
            #caculate the change of w(w0,w1,w2,w3) per step
            delta_w.append(learningRate*xi*(Y[i]-y))
        for j in range(len(w)):
            #update w
            w[j] = w[j]+delta_w[j]

print w
#---------------------------------------------------------------------------------------


########################################################################################
#--------------Third Method: Stochastic Gradient Decent--------------------------------------------#

learningRate = 0.001
epoch = 1000
#initialize the w0,w1,w2,w3
w = [0,1,1,1]

# Mini-batch gradient descent
# the second(above) method is equivalent to this method if batch_size = 1
# if batch_size = 1000, it's whole Batch gradient descent
batch_size = 50

for i in range(1000):
    # the fifth value of X[i] is y, so they can be shuffled together later
    X[i].append(Y[i])

for _ in range(epoch):
    #shuffle X at the begining of each epoch,
    #as the says above, the x0,x1,x2,x3 and y shuffled together
    np.random.shuffle(X)

    #batch loop
    for i in range(len(X)/batch_size):
        #distribute data to X_batch, the sizi of X_batch equals batch_size
        X_batch = X[i*batch_size:i*batch_size+batch_size]

        delta_w = [0,0,0,0]
        for j in range(batch_size):
            y = 0

            # caculate the predict:y
            for k in range(4):
                y = y + w[k]*X_batch[j][k]

            for k in range(4):
                #add each change of w per step during a batch
                #X_batch[j][-1] is the real value y of the data
                delta_w[k] = delta_w[k]+(learningRate*X_batch[j][k]*(X_batch[j][-1]-y))

        for j in range(4):
            #caculate the mean change of each w(w0,w1,w2,w3)
            delta_w[j] = delta_w[j]/batch_size

        for j in range(4):
            #update w
            w[j] = w[j]+delta_w[j]

print w

########################################################################################

