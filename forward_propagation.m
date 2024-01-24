function neural_network = forward_propagation(neural_network, x)
    
    %x: 1 x 196 oppure trainingset_size x 196
    
    Z_prev = x;

    for layer=1 : neural_network.total_layers_number
        neural_network.A{layer} = Z_prev * neural_network.W{layer}' + neural_network.B{layer};
        neural_network.Z{layer} = neural_network.activation_functions{layer}(neural_network.A{layer});
        Z_prev = neural_network.Z{layer};
    end
end

