addpath ./mnist/loadMnist
addpath ./error_functions
addpath ./activation_functions
addpath ./stop_criteria
addpath ./evaluation_functions

TRAININGSET_SIZE = 20000;
VALIDATIONSET_SIZE = 10000;
TESTSET_SIZE = 10000;

%IPERPARAMETRI
MAX_EPOCH = 1000;
HIDDEN_LAYERS_NUM_NEURONS = 100;
OUTPUT_LAYER_NUM_NEURONS = 10;

HIDDEN_ACTIVATION_FUNCTION = cell(1,length(HIDDEN_LAYERS_NUM_NEURONS));
HIDDEN_ACTIVATION_FUNCTION{1} = @sigmoid;

OUTPUT_ACTIVATION_FUNCTION = @identity;

ERROR_FUNCTION = @cross_entropy_soft_max;

STOP_CRITERION = @generalization_loss;

ETA_P = 1.05;
ETA_N = 0.65;

EARLYSTOPPING_ENABLED = 0;
ALPHA = 1;
STRIP = 5;

%{
    x_trainingset: trainingset_size x 196
    t_trainingset: trainingset_size x 10
    x_validationset: validationset_size x 196
    t_validationset: validationset_size x 10
    x_testset: testset_size x 196
    t_testset: testset_size x 10
%}

%Addestramento della rete neurale
[x_trainingset, t_trainingset, x_validationset, t_validationset, x_testset, t_testset] = import_mnist(TRAININGSET_SIZE, VALIDATIONSET_SIZE, TESTSET_SIZE);
neural_network = create_neural_network(size(x_testset,2), HIDDEN_LAYERS_NUM_NEURONS, OUTPUT_LAYER_NUM_NEURONS, HIDDEN_ACTIVATION_FUNCTION, OUTPUT_ACTIVATION_FUNCTION);
best_neural_network = learning_phase(neural_network, MAX_EPOCH, ETA_P, ETA_N, ERROR_FUNCTION, EARLYSTOPPING_ENABLED, STOP_CRITERION, ALPHA, STRIP, x_trainingset, t_trainingset, x_validationset, t_validationset);

%Valutazione della rete neurale
neural_network_test = forward_propagation(best_neural_network, x_testset);
acc = accuracy(neural_network_test.Z{end},t_testset);
fprintf("\nAccuracy: %s\n",num2str(acc));

