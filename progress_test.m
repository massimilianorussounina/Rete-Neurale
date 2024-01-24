addpath ./mnist/loadMnist
addpath ./error_functions
addpath ./activation_functions
addpath ./stop_criteria
addpath ./evaluation_functions

file_id = fopen("test_results/progress_results.csv","w");

fprintf("%s \t %s \t %s \t %s \t %s \t %s\n", "Neurons number", "Alpha", "Test Error", "Accuracy", "Best Epoch", "Stop Epoch");
fprintf(file_id, "%s \t %s \t %s \t %s \t %s \t %s\n", "Neurons number", "Alpha", "Test Error", "Accuracy", "Best Epoch", "Stop Epoch");

TEST = 5;

MAX_EPOCH = 1000;

HIDDEN_LAYER_NUM_NEURONS = 100;

ETA_P = 1.05;
ETA_N = 0.65;

OUTPUT_LAYER_NUM_NEURONS = 10;

HIDDEN_ACTIVATION_FUNCTION = cell(1,1);
HIDDEN_ACTIVATION_FUNCTION{1} = @sigmoid;


OUTPUT_ACTIVATION_FUNCTION = @identity;

ERROR_FUNCTION = @cross_entropy_soft_max;

STOP_CRITERION = @progress;

EARLYSTOPPING_ENABLED = 1;
ALPHA = [0.001, 0.01, 0.1, 0.5, 0.75, 1, 2, 3, 5];
STRIP = 5;

for i = 1 : length(HIDDEN_LAYER_NUM_NEURONS)
    for j = 1 : length(ALPHA)
        %{
        test_error = 0;
        acc = 0;
        best_epochs_avg = 0;
        stop_epoch_avg = 0;
        %}
        %for t = 1: TEST
            [x_trainingset, t_trainingset, x_validationset, t_validationset, x_testset, t_testset] = import_mnist(20000, 10000, 10000);
            neural_network = create_neural_network(size(x_testset,2), HIDDEN_LAYER_NUM_NEURONS(i), OUTPUT_LAYER_NUM_NEURONS, HIDDEN_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION);
            [best_neural_network,best_epoch,stop_epoch] = learning_phase(neural_network, MAX_EPOCH, ETA_P(i), ETA_N(i), ERROR_FUNCTION, EARLYSTOPPING_ENABLED, STOP_CRITERION, ALPHA(j), STRIP, x_trainingset, t_trainingset, x_validationset, t_validationset);
            neural_network_test = forward_propagation(best_neural_network, x_testset);
            %{
            test_error = test_error + ERROR_FUNCTION(best_neural_network.Z{end}, t_testset);
            acc = acc + accuracy(neural_network_test.Z{end},t_testset);
            best_epochs_avg = best_epochs_avg + best_epoch;
            stop_epoch_avg = stop_epoch_avg + stop_epoch;
            %}
            test_error = ERROR_FUNCTION(best_neural_network.Z{end}, t_testset);
            acc = accuracy(neural_network_test.Z{end},t_testset);
            best_epochs_avg = best_epoch;
            stop_epoch_avg = stop_epoch;
        %end
        %{
        test_error = test_error / TEST;
        acc = acc / TEST;
        best_epochs_avg = round(best_epochs_avg / TEST);
        stop_epoch_avg = round(stop_epoch_avg / TEST);
        %}
        fprintf("%s \t %s \t %s \t %s \t %s \t %s\n",num2str(HIDDEN_LAYER_NUM_NEURONS(i)),num2str(ALPHA(j)),num2str(test_error),num2str(acc),num2str(best_epochs_avg),num2str(stop_epoch_avg));
        fprintf(file_id,"%s \t %s \t %s \t %s \t %s \t %s\n",num2str(HIDDEN_LAYER_NUM_NEURONS(i)),num2str(ALPHA(j)),num2str(test_error),num2str(acc),num2str(best_epochs_avg),num2str(stop_epoch_avg));
    end
end


fclose(file_id);