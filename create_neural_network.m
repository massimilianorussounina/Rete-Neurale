function neural_network = create_neural_network(dim_input, hidden_layers_neurons_numbers, output_layer_neurons_number, hidden_activation_functions, output_activation_function)

    %{
    dim_input: trainingset_size
    hidden_layers_neurons_numbers: 1 x numero di strati interni (ognuno indica il numero di neuroni dell'i-esimo strato)
    output_layer_neurons_number: 10
    hidden_activation_functions: 1 x numero di strati interni (ognuno indica l'header della funzione di attivazione dell'i-esimo strato interno)
    output_activation_function: header della funzione di attivazione di output
    %}

    %CONTROLLI SULL'INPUT
    if (dim_input<=0)
        error("La dimensione dell'input deve essere >0");
    end
    
    hidden_layers_number=length(hidden_layers_neurons_numbers);
    total_layers_number = hidden_layers_number + 1;
    
    if hidden_layers_number==0
        error("Il numero di strati interni deve essere maggiore di 0");
    end
    
    for i=1:hidden_layers_number
        if (hidden_layers_neurons_numbers(i)<=0)
            error("Il numero di nodi di uno strato deve essere maggiore di 0");
        end
    end
    
    if isempty(output_layer_neurons_number)
        error("Deve esistere un unico strato di output");
    end
    
    if (hidden_layers_number ~= size(hidden_activation_functions,2))
        error("Il numero di funzioni di attivazione degli strati interni deve essere uguale al numero di strati interni");
    end
    
    
    %{
    W: array di matrici ognuna di dimensione numero di neuroni dello strato corrente x numero di neuroni dello strato precedente
    B: array di matrici ognuna di dimensione 1 x numero di neuroni dello strato corrente
    %}

    W = cell(1,total_layers_number);
    B = cell(1,total_layers_number);
    
    %{
    A: array di matrici in cui ogni matrice 1 x numero di neuroni dello strato corrente oppure trainingset_size x numero di neuroni dello strato corrente
    Z: array di matrici in cui ogni matrice 1 x numero di neuroni dello strato corrente oppure trainingset_size x numero di neuroni dello strato corrente
    %}

    A = cell(1,total_layers_number);
    Z = cell(1,total_layers_number);

    activation_functions = cell(1, total_layers_number);
    
    %rng(1000);
    rng("default");

    prev_layer_neurons_number = dim_input;
    for layer = 1: hidden_layers_number
        current_layer_neurons_number = hidden_layers_neurons_numbers(layer);
        W{layer} = rand(current_layer_neurons_number, prev_layer_neurons_number) - 0.5;
        B{layer} = rand(1, current_layer_neurons_number) - 0.5;
        activation_functions{layer} = hidden_activation_functions{layer};
        prev_layer_neurons_number = current_layer_neurons_number;
    end
    W{end} = rand(output_layer_neurons_number,prev_layer_neurons_number) - 0.5;
    B{end} = rand(1, output_layer_neurons_number) - 0.5;
    activation_functions{end} = output_activation_function;


    neural_network.total_layers_number = total_layers_number;
    neural_network.A = A;
    neural_network.Z = Z;
    neural_network.W = W;
    neural_network.B = B;
    neural_network.activation_functions = activation_functions;
end

