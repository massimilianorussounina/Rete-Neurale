function [neural_network,old_rprop] = resilient_backpropagation(neural_network, old_rprop, eta_p, eta_n, W_derivatives,B_derivatives)

    %{
    W_derivatives: array di matrici in cui ogni matrice è numero di neuroni dello strato corrente x numero di neuroni dello strato precedente
    B_derivatives: array di matrici in cui ogni matrice è 1 x numero di neuroni dello strato precedente
    %}
    
    DELTA_MAX = 50;
    DELTA_MIN = 1e-6;
    
    rng("default");

    %Si distingue la prima esecuzione dello RPROP dalle successive per l'inizializzazione degli step
    
    if (not(isfield(old_rprop,'W_derivatives')))
    
        old_rprop.W_derivatives = W_derivatives;
        old_rprop.B_derivatives = B_derivatives;
        
        %{
        W_step: array di matrici in cui ogni matrice è numero di neuroni dello strato corrente x numero di neuroni dello strato precedente
        B_step: array di matrici in cui ogni matrice è 1 x numero di neuroni dello strato precedente
        %}

        old_rprop.W_step = cell(1,neural_network.total_layers_number);
        old_rprop.B_step = cell(1,neural_network.total_layers_number);
        
        
        for layer = 1 : neural_network.total_layers_number
            %Calcolo degli step
            old_rprop.W_step{layer} = (0.2-0.05) .* rand(size(neural_network.W{layer})) + 0.05;
            old_rprop.B_step{layer} = (0.2-0.05) .* rand(size(neural_network.B{layer})) + 0.05;
            %Aggiornamento dei pesi
            neural_network.W{layer} = neural_network.W{layer} - old_rprop.W_step{layer} .* sign(W_derivatives{layer});
            neural_network.B{layer} = neural_network.B{layer} - old_rprop.B_step{layer} .* sign(B_derivatives{layer});
        end
        
    else
        %Aggiornamento dei pesi
        for layer = 1 : neural_network.total_layers_number
            
            %Vengono salvate le derivate e gli step precedenti nella struttura old_rprop

            epoch_gradient_product_W = W_derivatives{layer} .* old_rprop.W_derivatives{layer};
            epoch_gradient_product_B = B_derivatives{layer} .* old_rprop.B_derivatives{layer};
            old_rprop.W_step{layer} = min(old_rprop.W_step{layer} .* (eta_p .^ (epoch_gradient_product_W>0)),DELTA_MAX);
            old_rprop.W_step{layer} = max(old_rprop.W_step{layer} .* (eta_n .^ (epoch_gradient_product_W<0)),DELTA_MIN);
            old_rprop.B_step{layer} = min(old_rprop.B_step{layer} .* (eta_p .^ (epoch_gradient_product_B>0)),DELTA_MAX);
            old_rprop.B_step{layer} = max(old_rprop.B_step{layer} .* (eta_n .^ (epoch_gradient_product_B<0)),DELTA_MIN);

            %Aggiornamento dei pesi

            neural_network.W{layer} = neural_network.W{layer} - old_rprop.W_step{layer} .* sign(W_derivatives{layer});
            neural_network.B{layer} = neural_network.B{layer} - old_rprop.B_step{layer} .* sign(B_derivatives{layer});
        end
    end
    
end

