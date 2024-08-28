def write_variables_to_file(model, weights, biases, hidden_vars, binary_v_output, relu_activation, y_pred, hidden_layers, output_dim, n, filename):
    with open(filename, 'w') as f:
        # Write the values of weight variables
        for l in range(len(weights)):
            W = weights[l]
            #abs_W = abs_weights[l]
            for key in W.keys():
                f.write(f"Weight W{l+1}[{key}] = {W[key].X}\n")
                #f.write(f"abs_Weight abs_W{l+1}[{key}] = {abs_W[key].X}\n")


        # Write the values of bias variables
        for l in range(len(biases)):
            b = biases[l]
            #abs_b = abs_biases[l]
            for key in b.keys():
                f.write(f"Bias b{l+1}[{key}] = {b[key].X}\n")
                #f.write(f"abs_Bias abs_b{l+1}[{key}] = {abs_b[key].X}\n")

        # Write the values of auxiliary variables for prediction
        for i in range(n):
            for k in range(len(hidden_layers)):
                for j in range(hidden_layers[k]):
                    f.write(f"Auxiliary Variable for calculation of z = Wx + b [{i}, {j}] = {hidden_vars[k][i, j].X}\n")
                    f.write(f"Variable for hidden layer relu_activation[{i}, {j}] = {relu_activation[k][i, j].X}\n")

        # Write the values of auxiliary variables for prediction
        for i in range(n):
            for j in range(output_dim):
                f.write(f"Auxiliary Variable for calculation of z = Wx + b[{i}, {j}] = {hidden_vars[-1][i, j].X}\n")
                f.write(f"Binary_variable associated = {binary_v_output[i,j].X}\n")
                f.write(f"Prediction Variable y_pred[{i}, {j}] = {y_pred[i, j].X}\n")
        '''
        # Write the values of hinge loss terms
        for i in range(n):
            for j in range(output_dim):
                f.write(f"Hinge Loss Term hinge_loss_terms[{i}, {j}] = {hinge_loss_terms[i, j].X}\n")
        
        # Write the values of correct_pred variables
        for i in range(n):
            #for j in range(output_dim):
                f.write(f"correct prediction for sample {i} = {correct_preds[i].X}\n")
        
        for i in range(n):
            for j in range(output_dim):
                f.write(f"Predicted class sample {i} class {j} = {predicted_class[i, j].X}\n")
        '''
