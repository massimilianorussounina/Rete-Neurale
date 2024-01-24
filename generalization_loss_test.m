addpath ./mnist/loadMnist
addpath ./error_functions
addpath ./activation_functions
addpath ./stop_criteria
addpath ./evaluation_functions

file_id = fopen("test_results/generalization_loss_results.csv","w");

fprintf("%s \t %s \t %s \t %s \t %s \t %s\n", "Neurons number", "Alpha", "Test Error", "Accuracy", "Best Epoch", "Stop Epoch");
fprintf(file_id, "%s \t %s \t %s \t %s \t %s \t %s\n", "Neurons number", "Alpha", "Test Error", "Accuracy", "Best Epoch", "Stop Epoch");

TRAININGSET_SIZE = 20000;
VALIDATIONSET_SIZE = 10000;
TESTSET_SIZE = 10000;

MAX_EPOCH = 1000;

HIDDEN_LAYER_NUM_NEURONS = 100;

ETA_P = 1.05;
ETA_N = 0.65;

OUTPUT_LAYER_NUM_NEURONS = 10;

HIDDEN_ACTIVATION_FUNCTION = cell(1,1);
HIDDEN_ACTIVATION_FUNCTION{1} = @sigmoid;


OUTPUT_ACTIVATION_FUNCTION = @identity;

ERROR_FUNCTION = @cross_entropy_soft_max;

STOP_CRITERION = @generalization_loss;

EARLYSTOPPING_ENABLED = 1;
ALPHA = [0.01, 0.1, 0.5, 1, 3, 5];
STRIP = 5; %not used

for i = 1 : length(HIDDEN_LAYER_NUM_NEURONS)
    for j = 1 : length(ALPHA)
            [x_trainingset, t_trainingset, x_validationset, t_validationset, x_testset, t_testset] = import_mnist(TRAININGSET_SIZE, VALIDATIONSET_SIZE, TESTSET_SIZE);
            neural_network = create_neural_network(size(x_testset,2), HIDDEN_LAYER_NUM_NEURONS(i), OUTPUT_LAYER_NUM_NEURONS, HIDDEN_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION);
            [best_neural_network,best_epoch,stop_epoch] = learning_phase(neural_network, MAX_EPOCH, ETA_P(i), ETA_N(i), ERROR_FUNCTION, EARLYSTOPPING_ENABLED,STOP_CRITERION, ALPHA(j), STRIP, x_trainingset, t_trainingset, x_validationset, t_validationset);
            neural_network_test = forward_propagation(best_neural_network, x_testset);
            test_error = ERROR_FUNCTION(best_neural_network.Z{end}, t_testset);
            acc = accuracy(neural_network_test.Z{end},t_testset);
            fprintf("%s \t %s \t %s \t %s \t %s \t %s\n",num2str(HIDDEN_LAYER_NUM_NEURONS(i)),num2str(ALPHA(j)),num2str(test_error),num2str(acc),num2str(best_epoch),num2str(stop_epoch));
            fprintf(file_id, "%s \t %s \t %s \t %s \t %s \t %s\n",num2str(HIDDEN_LAYER_NUM_NEURONS(i)),num2str(ALPHA(j)),num2str(test_error),num2str(acc),num2str(best_epoch),num2str(stop_epoch));
    end
end

fclose(file_id);