import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
import sys
import os

def scaling(input_x):
    for col in input_x.columns:
        std = input_x[col].std()
        if std!=0:
            input_x[col] = (input_x[col] - input_x[col].mean()) / std
        else:
            input_x[col] = (input_x[col] - input_x[col].mean())
            
    return input_x

def norms_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix,axis=0)
    for i in range(len(norms)):
        if norms[i] == 0:
            norms[i] =1
    return feature_matrix/norms, norms

def get_features1(file_path):
	# Given a file path , return feature matrix and target labels 
    data = pd.read_csv(file_path)
    data['post_day'] = data['post_day'].astype('category')
    data['basetime_day'] = data['basetime_day'].astype('category')
    data['post_day'] = data['post_day'].cat.codes
    data['basetime_day'] = data['basetime_day'].cat.codes
    data[data.columns[0:-1]] = scaling(data[data.columns[0:-1]])
    print(data[data.columns[1]])
    phi = np.array(data[data.columns[0:-1]])
    y=np.array(data[data.columns[-1]])
    phi=np.concatenate((np.ones(phi.shape[0])[:, np.newaxis], phi), axis=1)
    y=y.reshape(-1,1)
    return phi, y

def get_features2(file_path):
    data = pd.read_csv(file_path)
    orig_data = data
    data= data[data.columns[0:-1]]
    onehot1 = pd.get_dummies(data["post_day"],prefix="post_day")
    onehot2 = pd.get_dummies(data["basetime_day"],prefix="basetime_day")
    data=data.join(onehot1)
    data=data.join(onehot2)
    data.drop(["post_day"],axis=1,inplace=True)
    data.drop(["basetime_day"],axis=1,inplace=True)
    y = orig_data.target
    y=np.array(y).reshape(-1,1)
    return data,y

def best_features1(w_final):
    x= dict(enumerate(np.abs(w_final[1:])))
    final_w =sorted(x.items(), key=lambda kv: kv[1], reverse=True)
    indexes= dict(np.array(final_w[0:10]))
    index=list(indexes.keys())
    return index

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    feature_matrix, norms = norms_features(feature_matrix)
    weights=weights.reshape(-1,1)
    output=np.array(output).reshape(-1,1)
    prediction = predict_output(feature_matrix,weights)
    prediction = prediction.reshape(-1,1)
    error=output-prediction
    feature_i = feature_matrix[:,i].reshape(-1,1)
    temp= weights[i] * feature_i 
    ro_i = (np.sum(feature_i * (error+ temp)))
    if ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2.
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2.
    else:
        new_weight_i = 0.
    
    return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    changed = False
    iterations = 1
    initial_weights = initial_weights.reshape(-1,1)
    while not changed:
        difference=[]
        for i in range(len(initial_weights)):
            old_weights_i = initial_weights[i]
            initial_weights[i]= lasso_coordinate_descent_step(i, feature_matrix, output, initial_weights,l1_penalty)
            difference.append(initial_weights[i] - old_weights_i) 
        if max(difference) < tolerance:
            changed = True
        
        iterations += 1
        if iterations%1000 == 0:
            print(iterations)
            print("Max Diff",max(difference))
            
    weights= initial_weights
    return weights


def predict_output(phi,w):
    output=np.ceil(np.dot(phi,w))
    return output.reshape(output.shape[0],1)

def compute_error(phi, y , w) :
    y_pred = predict_output(phi,w)
    error = np.average(np.square(y_pred - y))
    rmse = np.sqrt(error)
    return error, rmse

def task2_compute_error(x_train2, y_train2,w_task2):
    x_train2=np.concatenate((np.ones(x_train2.shape[0])[:, np.newaxis], x_train2), axis=1)
    feature_matrix_norm, norms = norms_features(x_train2)
    norms=norms.reshape(-1,1)
    w_task2_norm = w_task2/norms
    w_task2_norm= w_task2_norm.reshape(-1,1)
    err_task2, rmse = compute_error(feature_matrix_norm,y_train2,w_task2_norm)
    np.save("feature_matrix_norm.npy", feature_matrix_norm)
    np.save("w_task2.npy", w_task2_norm)
    return err_task2, rmse, w_task2_norm
    

def calc_grad(error,phi_i,w_i,l2_penalty,is_bias):
    phi_i=phi_i.reshape(-1,1)
    
    if is_bias == True:
        return (2*np.sum(error*phi_i))/len(error) 
    else:
        return (2*np.sum(error*phi_i) + 2*l2_penalty*w_i)/len(error)
    
def calc_grad_p(error,phi_i,w_i,lp_penalty,is_bias,p):
    phi_i=phi_i.reshape(-1,1)
    
    if is_bias == True:
        return (2*np.sum(error*phi_i))/len(error) 
    else:
        return ((2*np.sum(error*phi_i)) + lp_penalty*(w_i**(p-1)))/len(error)
    
def task1(phi, y):
    w = np.zeros(phi.shape[1])
    w = w.reshape(-1,1)
    alpha = 1.25e-4
    l2_penalty= 0
    tolerance =1e-5
    iterations=1
    
    while (True):
        prediction = predict_output(phi,w)
        error = prediction - y
        w_up = w.copy()
        for i in range(len(w)):
            if i==0:
                grad = calc_grad(error,phi[:,i],w[i],l2_penalty,True)
            else:
                grad = calc_grad(error,phi[:,i],w[i],l2_penalty,False)
            
            w[i]-= alpha * grad

        if iterations % 1000 == 0:
            print("Iteration: %d - Error: %.4f",iterations, np.sum(abs(w_up - w)))
        iterations += 1
        
        if np.sum(abs(w_up - w)) < tolerance:
            break

	#2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i].
    w_final=w
    return w_final


def task2(phi, y,plot=False):
    """
     generates and saves plots of top three features with target variable.
     Note: Procedure to obtain top features is important
    """
    
    orig_data = phi.copy()
    l1_penalty =100
    tolerance = 1e-3
    phi=np.concatenate((np.ones(phi.shape[0])[:, np.newaxis], phi), axis=1)
    w_task2= lasso_cyclical_coordinate_descent(np.array(phi), y, np.zeros(phi.shape[1]), l1_penalty,tolerance)
    w_task2=w_task2.reshape(-1,1)
    index= best_features1(w_task2)
    index = [int(i) for i in index]
    headers = list(orig_data[orig_data.columns[index]].dtypes.index)
    
    if plot:
        plt.figure(1)
        plt.subplot(311)
        plt.scatter(orig_data[orig_data.columns[int(index[0])]], y, c='r',s=5)
        plt.xlabel(headers[0])
        plt.ylabel("TARGET")
        plt.xlim(0, 1500)
        plt.ylim(0, 1000)
        plt.savefig("Plot1")
        plt.close()


        plt.subplot(312)
        plt.scatter(orig_data[orig_data.columns[int(index[1])]],y, c='b', s=5)
        plt.xlabel(headers[1])
        plt.ylabel("TARGET")
        plt.xlim(0, 75)
        plt.ylim(0, 1000)
        plt.savefig("Plot2")
        plt.close()


        plt.subplot(313)
        plt.scatter(orig_data[orig_data.columns[int(index[2])]], y, c='g', s=5)
        plt.xlabel(headers[2])
        plt.ylabel("TARGET")
        plt.xlim(0, 1000)
        plt.ylim(0, 200)
        plt.savefig("Plot3")
        plt.close()

    return w_task2
   
def task3(phi, y , lamda, p):
    w = np.zeros(phi.shape[1])
    w = w.reshape(-1,1)
    alpha = 1.25e-4
    tolerance =1e-5
    iterations=1
    
    while (True):
        prediction = predict_output(phi,w)
        error = prediction - y
        w_up = w.copy()
        for i in range(len(w)):
            if i==0:
                grad = calc_grad_p(error,phi[:,i],w[i],lamda,True,p)
            else:
                grad = calc_grad_p(error,phi[:,i],w[i],lamda,False,p)
            
            w[i]-= alpha * grad
            
        if iterations % 1000 == 0:
            print("Iteration: %d - Error: %.4f",iterations, np.sum(abs(w_up - w)))
        iterations += 1
        if np.sum(abs(w_up - w)) < tolerance:
            break

    w_task3=w
    return w_task3

def basis1(new_data,y,index,test_file,output_file):

    phi_test = pd.read_csv(test_file)
    onehot1 = pd.get_dummies(phi_test["post_day"],prefix="post_day")
    onehot2 = pd.get_dummies(phi_test["basetime_day"],prefix="basetime_day")
    phi_test=phi_test.join(onehot1)
    phi_test=phi_test.join(onehot2)
    phi_test.drop(["post_day"],axis=1,inplace=True)
    phi_test.drop(["basetime_day"],axis=1,inplace=True)

    new_index = []
    for i in index:
        new_index.append(int(i))
   
    new = new_data[new_data.columns[new_index]].copy()
    for col in list(new.columns):
        new_data[col] = new[col].apply(lambda x: log(x) if int(x)>0 else 0)
        phi_test[col] = phi_test[col].apply(lambda x: log(x) if int(x)>0 else 0)
    
    phi_test=np.concatenate((np.ones(phi_test.shape[0])[:, np.newaxis], phi_test), axis=1)
    phi_test, norms_test= norms_features(phi_test)
    
    w_task_basis1=task2(new_data,y,False)
    w_task_basis1 = w_task_basis1.reshape(-1,1)
    
    new_data=np.concatenate((np.ones(new_data.shape[0])[:, np.newaxis], new_data), axis=1)
    new_data, norms= norms_features(new_data)
    norms = norms.reshape(-1,1)
    w_task_basis1 = w_task_basis1/norms
    pred=predict_output(new_data,w_task_basis1)
    
    for i in range(pred.shape[0]):
            if pred[i] < 0:
                pred[i] = 0
    error= np.sum(np.square(y-pred))/new_data.shape[0]
    print("MSE=  ",error)
    print("RMSE= ",np.sqrt(error))
    
    

    out=predict_output(phi_test,w_task_basis1)
    for i in range(out.shape[0]):
        if out[i] < 0:
            out[i] = 0
   
    output = pd.DataFrame(out)
    output.to_csv(output_file,header=["target"],index_label="Id")
    
    return w_task_basis1

def basis2(new_data,y,index,test_file,output_file):

    phi_test = pd.read_csv(test_file)
    onehot1 = pd.get_dummies(phi_test["post_day"],prefix="post_day")
    onehot2 = pd.get_dummies(phi_test["basetime_day"],prefix="basetime_day")
    phi_test=phi_test.join(onehot1)
    phi_test=phi_test.join(onehot2)
    phi_test.drop(["post_day"],axis=1,inplace=True)
    phi_test.drop(["basetime_day"],axis=1,inplace=True)

    new_data["c2_square"] = new_data["c2"].apply(lambda x: x**2)
    phi_test["c2_square"] = phi_test["c2"].apply(lambda x: x**2)
    new_data["c2_cube"] = new_data["c2"].apply(lambda x: x**3)
    phi_test["c2_cube"] = phi_test["c2"].apply(lambda x: x**3)
    new_data["c2_4"] = new_data["c2"].apply(lambda x: x**4)
    phi_test["c2_4"] = phi_test["c2"].apply(lambda x: x**4)
    new_data["c2_5"] = new_data["c2"].apply(lambda x: x**5)
    phi_test["c2_5"] = phi_test["c2"].apply(lambda x: x**5)
    new_data["share_count_log"] = new_data["share_count"].apply(lambda x: log(x) if int(x)>1 else 0)
    phi_test["share_count_log"] = phi_test["share_count"].apply(lambda x: log(x) if int(x)>1 else 0)
    new_data["c3_square"] = new_data["c3"].apply(lambda x: x**2)
    phi_test["c3_square"] = phi_test["c3"].apply(lambda x: x**2)
    new_data["c3_cube"] = new_data["c3"].apply(lambda x: x**3)
    phi_test["c3_cube"] = phi_test["c3"].apply(lambda x: x**3)
    new_data["c3_4"] = new_data["c3"].apply(lambda x: x**4)
    phi_test["c3_4"] = phi_test["c3"].apply(lambda x: x**4)
    new_data["c3_5"] = new_data["c3"].apply(lambda x: x**5)
    phi_test["c3_5"] = phi_test["c3"].apply(lambda x: x**5)
    new_data["base_time_square"] = new_data["base_time"].apply(lambda x: x**2)
    phi_test["base_time_square"] = phi_test["base_time"].apply(lambda x: x**2)
    new_data["base_time_cube"] = new_data["base_time"].apply(lambda x: x**3)
    phi_test["base_time_cube"] = phi_test["base_time"].apply(lambda x: x**3)
    new_data["base_time_4"] = new_data["base_time"].apply(lambda x: x**4)
    phi_test["base_time_4"] = phi_test["base_time"].apply(lambda x: x**4)
    new_data["base_time_5"] = new_data["base_time"].apply(lambda x: x**5)
    phi_test["base_time_5"] = phi_test["base_time"].apply(lambda x: x**5)
    new_data["page_likes_log"] = new_data["page_likes"].apply(lambda x: log(x) if int(x)>1 else 0)
    phi_test["page_likes_log"] = phi_test["page_likes"].apply(lambda x: log(x) if int(x)>1 else 0)
        
    phi_test=np.concatenate((np.ones(phi_test.shape[0])[:, np.newaxis], phi_test), axis=1)
    phi_test, norms_test= norms_features(phi_test)
    
    w_task_basis2=task2(new_data,y,False)
    w_task_basis2 = w_task_basis2.reshape(-1,1)
    
    new_data=np.concatenate((np.ones(new_data.shape[0])[:, np.newaxis], new_data), axis=1)
    new_data, norms= norms_features(new_data)
    norms = norms.reshape(-1,1)
    w_task_basis2 = w_task_basis2/norms
    pred=predict_output(new_data,w_task_basis2)
    
    for i in range(pred.shape[0]):
            if pred[i] < 0:
                pred[i] = 0
    error= np.sum(np.square(y-pred))/new_data.shape[0]
    print("MSE=  ",error)
    print("RMSE= ",np.sqrt(error))
    
    

    out=predict_output(phi_test,w_task_basis2)
    for i in range(out.shape[0]):
        if out[i] < 0:
            out[i] = 0
   
    output = pd.DataFrame(out)
    output.to_csv(output_file,header=["target"],index_label="Id")
    
    return w_task_basis2

def task4(train_file, test_file,basis):
    data = pd.read_csv(train_file)
    phi = data[data.columns[0:-1]]
    onehot1 = pd.get_dummies(phi["post_day"],prefix="post_day")
    onehot2 = pd.get_dummies(phi["basetime_day"],prefix="basetime_day")
    phi=phi.join(onehot1)
    phi=phi.join(onehot2)
    phi.drop(["post_day"],axis=1,inplace=True)
    phi.drop(["basetime_day"],axis=1,inplace=True)
    y=np.array(data[data.columns[-1]])
    y = y.reshape(-1,1)

    w_task4 = task2(phi,y,False)
    index = best_features1(w_task4)
    
    if basis==1:
        w_basis = basis1(phi,y,index,test_file,"BASIS1.csv")
    elif basis == 2:
        w_basis = basis2(phi,y,index,test_file,"BASIS2.csv")
    else:
        print("Invalid Choice")
        
    return w_basis
    

def task5(phi,y):
    l1_penalty =100
    tolerance = 1e-3
    phi=np.concatenate((np.ones(phi.shape[0])[:, np.newaxis], phi), axis=1)
    w_task5= lasso_cyclical_coordinate_descent(np.array(phi), y, np.zeros(phi.shape[1]), l1_penalty,tolerance)
    w_task5=w_task5.reshape(-1,1)
    
    return w_task5

def gen_output1(file_path,output_file_path, w_task):
    phi_test = pd.read_csv(file_path)
    phi_test['post_day'] = phi_test['post_day'].astype('category')
    phi_test['basetime_day'] = phi_test['basetime_day'].astype('category')
    phi_test['post_day'] = phi_test['post_day'].cat.codes
    phi_test['basetime_day'] = phi_test['basetime_day'].cat.codes
    phi_test=scaling(phi_test)
    phi_test=np.concatenate((np.ones(phi_test.shape[0])[:, np.newaxis], phi_test), axis=1)
    out=predict_output(phi_test,w_task)
    for i in range(out.shape[0]):
            if out[i] < 0:
                out[i] = 0
    output = pd.DataFrame(out)
    output.to_csv(output_file_path,header=["target"],index_label="Id")
    
    

def gen_output2(file_path,output_file_path,w_task):
    phi_test = pd.read_csv(file_path)
    onehot1 = pd.get_dummies(phi_test["post_day"],prefix="post_day")
    onehot2 = pd.get_dummies(phi_test["basetime_day"],prefix="basetime_day")
    phi_test=phi_test.join(onehot1)
    phi_test=phi_test.join(onehot2)
    phi_test.drop(["post_day"],axis=1,inplace=True)
    phi_test.drop(["basetime_day"],axis=1,inplace=True)
    phi_test, norms= norms_features(phi_test)
    phi_test=np.concatenate((np.ones(phi_test.shape[0])[:, np.newaxis], phi_test), axis=1)
    out=predict_output(phi_test,w_task)
    for i in range(out.shape[0]):
            if out[i] < 0:
                out[i] = 0
    output = pd.DataFrame(out)
    output.to_csv(output_file_path,header=["target"],index_label="Id")

    
def main():
    """ 
    Calls functions required to do tasks in sequence 
    say : 
        train_file = first_argument
        test_file = second_argument
        x_train, y_train = get_features();
        task1();task2();task3();.....
    """
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
#     x_train1,y_train1 = get_features1(train_file)
#     w_task1 = task1(x_train1,y_train1)
#     err_task1 = compute_error(x_train1,y_train1,w_task1)
##    gen_output1(test_file,output_file_path, w_task1)
    
#     x_train2,y_train2 = get_features2(train_file)
#     if not os.path.exists("w_task2.npy"):
#         w_task2= task2(x_train2,y_train2, True)
#         err_task2, rmse, w_task2_norm = task2_compute_error(x_train2, y_train2,w_task2)
#         print("RMSE on Train:",rmse)
#     else:
#         feature_matrix_norm = np.load("feature_matrix_norm.npy")
#         w_task2_norm = np.load("w_task2.npy")
#         err_task2, rmse = compute_error(feature_matrix_norm,y_train2,w_task2_norm)
#         print("RMSE on Train:",rmse)
#     gen_output2("test.csv","Task2Pred.csv",w_task2_norm)

#     x_train3,y_train3 = get_features1(train_file)
#     p=4
#     w_task3 = task3(x_train3,y_train3,1000,p)
#     err_task3 = compute_error(x_train3,y_train3,w_task3)
#     p=6
#     w_task3 = task3(x_train3,y_train3,1000,p)
#     err_task3 = compute_error(x_train3,y_train3,w_task3)

#     w_task4 = task4(train_file, test_file,2) 

#     x_train5,y_train5 = get_features2(train_file)
#     w_task5 = task5(x_train5,y_train5)
#     err_task5, rmse, w_task2_norm = task2_compute_error(x_train5, y_train5,w_task5)
#     print("RMSE on Train:",rmse)
##    gen_output2(test_file,output_file_path, w_task5)
#     print("Errors:", err_task1,err_task2,err_task3,err_task5)

main()


