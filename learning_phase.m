function [best_neural_network, best_epoch, stop_epoch] = learning_phase(neural_network, max_epoch, eta_p, eta_n, error_function, earlystopping_enabled, stop_criterion_function, alpha, strip, x_trainingset, t_trainingset, x_validationset, t_validationset)
    
    %{
    x_trainingset: trainingset_size x 196
    t_trainingset: trainingset_size x 10
    x_validationset: validationset_size x 196
    t_validationset: validationset_size x 10
    %}

    training_error = zeros(1,max_epoch);
    validation_error = zeros(1,max_epoch);

    best_neural_network = neural_network;
    minimum_error = realmax;
    best_epoch = 0;
    
    %{
    Per disegnare i plot
    minimum_errors = [];
    minimum_errors_length = 1;
    best_epochs = [];
    best_epochs_length = 1;
    %}

    old_rprop = struct();
    
    neural_network = forward_propagation(neural_network,x_trainingset);

    %Apprendimento BATCH
    for current_epoch = 1: max_epoch
        [W_derivatives,B_derivatives] = backpropagation(neural_network, x_trainingset, t_trainingset, error_function);
        [neural_network,old_rprop] = resilient_backpropagation(neural_network, old_rprop, eta_p, eta_n, W_derivatives,B_derivatives);

        neural_network = forward_propagation(neural_network, x_trainingset);

        training_error(current_epoch) = error_function(neural_network.Z{end},t_trainingset);

        neural_network = forward_propagation(neural_network, x_validationset);

        validation_error(current_epoch) = error_function(neural_network.Z{end},t_validationset);
        
        if (validation_error(current_epoch) < minimum_error)
            minimum_error = validation_error(current_epoch);
            best_neural_network = neural_network;
            best_epoch = current_epoch;
            %{
            Per disegnare i plot
            minimum_errors(minimum_errors_length) = minimum_error;
            minimum_errors_length = minimum_errors_length + 1;
            best_epochs(best_epochs_length) = best_epoch;
            best_epochs_length = best_epochs_length + 1;
            %}
        end
        acc = accuracy(neural_network.Z{end},t_validationset);
        fprintf("Epoca: %s, Training Error: %s, Validation Error: %s, Minimum Error: %s, Best Epoch: %s, Accuracy: %s\n", num2str(current_epoch),num2str(training_error(current_epoch)),num2str(validation_error(current_epoch)),num2str(minimum_error),num2str(best_epoch),num2str(acc));
        
        %Criterio di early stopping
        if isequal(stop_criterion_function, @generalization_loss)
            stop_criterion = stop_criterion_function(current_epoch, validation_error, minimum_error);
        else
            stop_criterion = stop_criterion_function(current_epoch, training_error, validation_error, minimum_error, strip);
        end
        
        if earlystopping_enabled
            if stop_criterion > alpha
                break;
            end
        end
    end
    stop_epoch = current_epoch;
    


    %Disegno dei plot


    %{
    namefile = strcat(num2str(eta_p),"-",num2str(eta_n),".jpg");
    p = plot(1:current_epoch,validation_error(1:current_epoch));
    hold on
    plot(best_epochs,minimum_errors,'r*');
    plot(1:current_epoch,training_error(1:current_epoch),'k');
    hold off
    legend('Validation Error','Minimum Validation Error','Training Error');
    %}
    %saveas(p,namefile);
    
end

