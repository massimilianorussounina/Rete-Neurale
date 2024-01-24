function [W_derivatives, B_derivatives] = backpropagation(neural_network, x, t, error_function)
    
    %{
    x: 1 x 196 oppure trainingset_size x 196
    y: 1 x 10 oppure trainingset_size x 10
    %}

    addpath ./activation_functions
    
    %{
    W_derivatives: array di matrici in cui ogni matrice è numero di neuroni dello strato corrente x numero di neuroni dello strato precedente
    B_derivatives: array di matrici in cui ogni matrice è 1 x numero di neuroni dello strato precedente
    %}

    W_derivatives = cell(1,neural_network.total_layers_number);
    B_derivatives = cell(1,neural_network.total_layers_number);

    %delta: 1 x numero di neuroni(layer) oppure trainingset_size x numero di neuroni(layer)
    
    delta = cell(1,neural_network.total_layers_number);

    output_activation_function_derivative = str2func(strcat('@dev_',char(neural_network.activation_functions{end})));
    error_function_derivative = str2func(strcat('@dev_',char(error_function)));
    
    neural_network = forward_propagation(neural_network, x);

    %Calcolo dei delta_k (neuroni di output)
    delta{end} = output_activation_function_derivative(neural_network.A{end});
    delta{end} = delta{end} .* error_function_derivative(neural_network.Z{end},t);

    %Calcolo dei delta_i (neuroni interni)
    for layer = neural_network.total_layers_number -1 : -1: 1
        delta{layer} = delta{layer + 1} * neural_network.W{layer + 1};
        activation_function_derivate = str2func(strcat('@dev_',char(neural_network.activation_functions{layer})));
        delta{layer} = delta{layer} .* activation_function_derivate(neural_network.A{layer});
    end

    %Calcolo delle derivate
    Z = x;
    for layer = 1 : neural_network.total_layers_number
        W_derivatives{layer} = delta{layer}' * Z;
        B_derivatives{layer} = sum(delta{layer},1);
        Z = neural_network.Z{layer};
    end
end

