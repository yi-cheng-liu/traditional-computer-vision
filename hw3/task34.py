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
    print('S: ', ans[:4])
    print('t: ', ans[4:])
    
    # 3. transform the points
    # [x_trans, y_trans].T = A([x, y].T) + b
    x_trans, y_trans = np.dot(ans[:4].reshape(2, 2), np.vstack([x, y])) + ans[4:].reshape(2, 1)
    
    # 4. plot the original points and transformed points
    return x, y, x_prime, y_prime, x_trans, y_trans

def p4(data):
    # code for Task 4
    x = data[:, 0]
    y = data[:, 1]
    x_prime = data[:, 2]
    y_prime = data[:, 3]
    
    # 2. calcualte the Homography matrix, and dot with p matrix
    H = fit_homography(data)
    p_trans = np.vstack((x,y,np.ones(len(x)))).T
    p_transform = np.dot(H, p_trans.T)
    x_trans = p_transform[0, :]
    y_trans = p_transform[1, :]
    scalar = np.sqrt(x_prime ** 2 + y_prime ** 2) / np.sqrt(x_trans ** 2 + y_trans ** 2)
    x_trans *= scalar
    y_trans *= scalar
    print('H: ', H)
    
    return x, y, x_prime, y_prime, x_trans, y_trans

def plot_result(x, y, x_prime, y_prime, x_trans, y_trans, i):
    plt.figure()
    plt.title('case_'+str(i))
    plt.scatter(x, y, color='blue', label='data_'+str(i))
    plt.scatter(x_prime, y_prime, color='green', label='data_prime_'+str(i))
    plt.scatter(x_trans, y_trans, color='red', label='data_trans_'+str(i))
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    # Task 3
    # 1. load the data
    data_1 = np.load('task3/points_case_1.npy')
    data_2 = np.load('task3/points_case_2.npy')
    
    # 2. calculate the points 
    print('case1')
    x_1, y_1, x_1_prime, y_1_prime, x_1_trans, y_1_trans = p3(data_1)
    print('case2')
    x_2, y_2, x_2_prime, y_2_prime, x_2_trans, y_2_trans = p3(data_2)
    
    # 3. plot the diagram
    plot_result(x_1, y_1, x_1_prime, y_1_prime, x_1_trans, y_1_trans, 1)
    plot_result(x_2, y_2, x_2_prime, y_2_prime, x_2_trans, y_2_trans, 2)
    
    # Task 4
    # 1. read the data and extract the x, y, x', y'
    data_1 = np.load('task4/points_case_1.npy')
    data_4 = np.load('task4/points_case_4.npy')
    data_5 = np.load('task4/points_case_5.npy')
    data_9 = np.load('task4/points_case_9.npy')
    
    # 2. calculate the points 
    print('case1')
    x_1, y_1, x_1_prime, y_1_prime, x_1_trans, y_1_trans = p4(data_1)
    print('case4')
    x_4, y_4, x_4_prime, y_4_prime, x_4_trans, y_4_trans = p4(data_4)
    print('case5')
    x_5, y_5, x_5_prime, y_5_prime, x_5_trans, y_5_trans = p4(data_5)
    
    print('case9')
    x_9, y_9, x_9_prime, y_9_prime, x_9_trans, y_9_trans = p4(data_9)
    
    # 3. plot the diagram
    plot_result(x_5, y_5, x_5_prime, y_5_prime, x_5_trans, y_5_trans, 5)
    plot_result(x_9, y_9, x_9_prime, y_9_prime, x_9_trans, y_9_trans, 9)