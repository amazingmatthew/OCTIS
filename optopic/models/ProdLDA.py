from optopic.models.pytorchavitm.AVITM import AVITM


class ProdLDA(AVITM):
    def __init__(self, num_topics=10, activation='softplus', dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3,
                 momentum=0.99, solver='adam', num_epochs=100, reduce_on_plateau=False, prior_mean=0.0,
                 prior_variance=None, num_layers=2, num_neurons=100):
        super(AVITM).__init__(num_topics=num_topics, model_type='prodLDA', activation=activation, dropout=dropout,
                              learn_priors=learn_priors, batch_size=batch_size, lr=lr, momentum=momentum,
                              solver=solver, num_epochs=num_epochs, reduce_on_plateau=reduce_on_plateau,
                              prior_mean=prior_mean, prior_variance=prior_variance, num_layers=num_layers,
                              num_neurons=num_neurons)

    def train_model(self, dataset, hyperparameters, top_words=10):
        super(AVITM).train_model(dataset, hyperparameters, top_words)
