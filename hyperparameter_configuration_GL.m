addpath ./mnist/loadMnist
addpath ./error_functions
addpath ./activation_functions
addpath ./stop_criteria
addpath ./evaluation_functions

file_id = fopen("test_results/hyperparameters_configuration_GL.csv","w");

fprintf("%s \t %s \t %s \t %s \t %s \t %s \t %s\n","Hidden Neurons", "Alpha", "Positive Eta", "Negative Eta", "Best Epoch", "Stop Epoch", "Accuracy");
fprintf(file_id, "%s \t %s \t %s \t %s \t %s \t %s \t %s\n","Hidden Neurons", "Alpha", "Positive Eta", "Negative Eta", "Best Epoch", "Stop Epoch", "Accuracy");
MAX_EPOCH = 1000;
HIDDEN_LAYER_NUM_NEURONS = [20, 40, 60, 80, 100];
ETA_P = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3];
ETA_N = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65];
OUTPUT_LAYER_NUM_NEURONS = 10;
HIDDEN_ACTIVATION_FUNCTION = cell(1,1);
HIDDEN_ACTIVATION_FUNCTION{1} = @sigmoid;


OUTPUT_ACTIVATION_FUNCTION = @identity;

ERROR_FUNCTION = @cross_entropy_soft_max;

STOP_CRITERION = @generalization_loss;

EARLYSTOPPING_ENABLED = 1;
ALPHA = [0.01, 0.1, 0.5, 1, 3, 5];
STRIP = 5; %not used

HIDDEN_LAYER_NUM_NEURONS_length = length(HIDDEN_LAYER_NUM_NEURONS);
ETA_P_length = length(ETA_P);
ETA_N_length = length(ETA_N);
ALPHA_length = length(ALPHA);


acc = 0;
for i = 1 : HIDDEN_LAYER_NUM_NEURONS_length
    for h = 1 : ALPHA_length
        for j = 1 : ETA_P_length
            for k = 1 : ETA_N_length
                [x_trainingset, t_trainingset, x_validationset, t_validationset, x_testset, t_testset] = import_mnist(20000, 10000, 10000);
                neural_network = create_neural_network(size(x_testset,2), HIDDEN_LAYER_NUM_NEURONS(i), OUTPUT_LAYER_NUM_NEURONS, HIDDEN_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION);
                [best_neural_network,best_epoch,stop_epoch] = learning_phase(neural_network, MAX_EPOCH, ETA_P(j), ETA_N(k), ERROR_FUNCTION, EARLYSTOPPING_ENABLED, STOP_CRITERION, ALPHA(h), STRIP, x_trainingset, t_trainingset, x_validationset, t_validationset);
                neural_network_test = forward_propagation(best_neural_network, x_testset);
                acc = accuracy(neural_network_test.Z{end},t_testset);
                fprintf("%s \t %s \t %s \t %s \t %s \t %s \t %s\n",num2str(HIDDEN_LAYER_NUM_NEURONS(i)),num2str(ALPHA(h)),num2str(ETA_P(j)),num2str(ETA_N(k)),num2str(best_epoch),num2str(stop_epoch),num2str(acc));
                fprintf(file_id, "%s \t %s \t %s \t %s \t %s \t %s \t %s\n",num2str(HIDDEN_LAYER_NUM_NEURONS(i)),num2str(ALPHA(h)),num2str(ETA_P(j)),num2str(ETA_N(k)),num2str(best_epoch),num2str(stop_epoch),num2str(acc));
            end
        end
    end
end

fclose(file_id);