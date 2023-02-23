import numpy as np
from matplotlib import pyplot as plt
from homography import fit_homography, homography_transform

def p3(data):
    # code for Task 3
    # 1. load points X from task3/
    x = np.transpose(data[:, :2])
    y = np.transpose(data[:, 2:])
    # Split data into x, y, x', y' arrays
    x = data[:, 0]
    y = data[:, 1]
    x_prime = data[:, 2]
    y_prime = data[:, 3]
    # Construct A and b matrices
    A = np.zeros((2*len(x), 6)) # (2n, 6)
    b = np.zeros(2*len(x))      # (2n, )
    for i in range(len(x)):
        A[2*i, :]   = [x[i], y[i],    0,    0, 1, 0]
        A[2*i+1, :] = [   0,    0, x[i], y[i], 0, 1]
        b[2*i] = x_prime[i]
        b[2*i+1] = y_prime[i]
    
    # 2. fit a transformation y=Sx+t
    ans = np.linalg.lstsq(A, b, rcond=None)[0]
    print('ans: ', ans)
    
    # 3. transform the points
    # [x_trans, y_trans].T = A([x, y].T) + b
    x_trans, y_trans = np.dot(ans[:4].reshape(2, 2), np.vstack([x, y])) + ans[4:].reshape(2, 1)
    
    # 4. plot the original points and transformed points
    return x, y, x_prime, y_prime, x_trans, y_trans

def p4():
    # code for Task 4

    pass

if __name__ == "__main__":
    # Task 3
    # 1. load the data
    data_1 = np.load('task3/points_case_1.npy')
    data_2 = np.load('task3/points_case_2.npy')
    
    # 2. calculate the points 
    x_1, y_1, x_1_prime, y_1_prime, x_1_trans, y_1_trans = p3(data_1)
    x_2, y_2, x_2_prime, y_2_prime, x_2_trans, y_2_trans = p3(data_2)
    
    # 3. plot the diagram
    plt.figure(1)
    plt.scatter(x_1, y_1, color='blue', label='data')
    plt.scatter(x_1_prime, y_1_prime, color='green', label='data_prime')
    plt.scatter(x_1_trans, y_1_trans, color='red', label='data_trans_1')
    plt.legend()
    
    plt.figure(2)
    plt.scatter(x_2, y_2, color='blue', label='data')
    plt.scatter(x_2_prime, y_2_prime, color='green', label='data_prime')
    plt.scatter(x_2_trans, y_2_trans, color='red', label='data_trans_2')
    plt.legend()
    plt.show()
    
    # Task 4
    p4()
    # 1. read the data and extract the x, y, x', y'
    data = np.load('task4/points_case_9.npy')
    x = data[:, 0]
    y = data[:, 1]
    x_prime = data[:, 2]
    y_prime = data[:, 3]
    
    # 2. calcualte the Homography matrix, and dot with p matrix
    H = fit_homography(data)
    p_trans = np.vstack((x,y,np.ones(len(x)))).T
    p_transform = np.dot(H, p_trans.T)
    print('H: ', H)
    
    # 3. print out the results 
    plt.scatter(x, y, color='blue')
    plt.scatter(x_prime, y_prime, color='green')
    plt.scatter(p_transform[0, :], p_transform[1, :], color='red')
    plt.show()