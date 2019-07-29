import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_topK_neighbor_nodes(X, adjacency_matrix, topK):
    batch_size, num_nodes, num_features = X.size()

    adjacency_matrix, neighbor_indices = torch.topk(adjacency_matrix, topK, dim=2)

    offsets = torch.arange(batch_size).type(neighbor_indices.type())
    neighbor_indices = neighbor_indices.view(batch_size, num_nodes * topK)
    neighbor_indices = (offsets * num_nodes).view(-1, 1) + neighbor_indices

    X_topK = X.view(-1, num_features)[neighbor_indices.view(-1)]
    X_topK = X_topK.view(batch_size, num_nodes, topK, num_features)

    # Normalize adjacency_matrix
    adjacency_matrix = adjacency_matrix.div(
        adjacency_matrix.sum(dim=2, keepdim=True) + 1e-12)

    return X_topK, adjacency_matrix


class UpdateFunction(nn.Module):
    def __init__(self, num_features_in, num_features_msg, num_features, dropout):
        super(UpdateFunction, self).__init__()
        assert(num_features_in > 0)
        assert(num_features_msg >= 0)
        assert(num_features > 0)
        num_features_concat = num_features_in + num_features_msg
        self.layers = nn.Sequential(
            nn.Linear(num_features_concat, num_features, bias=False),
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=dropout),
            nn.LeakyReLU())

        self.num_features_out = num_features_in + num_features


    def forward(self, X_input, X_message=None):
        assert(X_input.dim() == 3)
        batch_size, num_nodes, _ = X_input.size()

        if X_message is not None:
            assert(X_message.dim() == 3)
            assert(X_input.size(0) == X_message.size(0))
            assert(X_input.size(1) == X_message.size(1))
            X_output = torch.cat(
                [X_input.view(batch_size * num_nodes, -1),
                 X_message.view(batch_size * num_nodes, -1)], dim=1)
        else:
            X_output = X_input.view(batch_size * num_nodes, -1)

        X_output = self.layers(X_output)
        X_output = X_output.view(batch_size, num_nodes, -1)
        X_output = F.normalize(X_output, p=2, dim=X_output.dim()-1, eps=1e-12)
        # Divide by sqrt(2) in order to make the L2-norm  of X_output_final
        # equal to 1 (since both X_input and X_output are L2-normalized,
        # without the division by sqrt(2), the L2-norm of X_output_final would
        # be equal to sqrt(2)).
        X_output_final = torch.cat([X_input, X_output], dim=2) / math.sqrt(2)

        return X_output_final


class UpdateFunctionFinal(nn.Module):
    def __init__(self, num_features_in, num_features_msg, num_features):
        super(UpdateFunctionFinal, self).__init__()
        assert(num_features_in > 0)
        assert(num_features_msg >= 0)
        assert(num_features > 0)
        num_features_concat = num_features_in + num_features_msg
        self.layers = nn.Linear(num_features_concat, num_features)

        self.num_features_out = num_features

    def forward(self, X_input, X_message=None):
        assert(X_input.dim() == 3)
        batch_size, num_nodes, _ = X_input.size()

        if X_message is not None:
            assert(X_message.dim() == 3)
            assert(X_input.size(0) == X_message.size(0))
            assert(X_input.size(1) == X_message.size(1))
            X_output = torch.cat(
                [X_input.view(batch_size * num_nodes, -1),
                 X_message.view(batch_size * num_nodes, -1)], dim=1)
        else:
            X_output = X_input.view(batch_size * num_nodes, -1)

        X_output = self.layers(X_output)
        X_output = X_output.view(batch_size, num_nodes, -1)

        return X_output


class RelationNetBasedAggregationFunction(nn.Module):
    def __init__(self, num_features_in, num_features_msg, dropout, topK):
        super(RelationNetBasedAggregationFunction, self).__init__()

        self.num_features_in = num_features_in
        self.num_features_msg = num_features_msg
        self.dropout = dropout
        self.topK = topK

        print('==> RelationNetBasedAggregationFunction Options:')
        print('====> num_features_in: {0}'.format(self.num_features_in))
        print('====> num_features_msg: {0}'.format(self.num_features_msg))
        print('====> dropout: {0}'.format(self.dropout))
        print('====> topK: {0}'.format(self.topK))

        self.linear = nn.Linear(num_features_in, num_features_msg)
        self.activation_function = nn.Sequential(
            nn.BatchNorm1d(num_features_msg),
            nn.Dropout(p=dropout),
            nn.LeakyReLU())

    def forward(self, X, adjacency_matrix):
        """
        Args:
            X_input: it is a 3-D tensor with shape
                [batch_size, num_nodes, num_features] that represents the input
                signal to the GNN layer.
            adjacency_matrix: a 4-D tensor with shape
                [batch_size, num_nodes, num_nodes, 1] that
                represents the set of adjacency matrices used for each batch
                item. For example adjacency_matrix[b, :, :, i] is the i-th
                adjacency matrix of of the b-th batch item. num_nodes is
                the number of nodes in the graph.
        """
        assert(adjacency_matrix.size(0) == X.size(0))
        assert(adjacency_matrix.size(1) == X.size(1))
        assert(adjacency_matrix.size(2) == X.size(1))
        assert(adjacency_matrix.size(3) == 1)


        batch_size, num_nodes, _, _ = adjacency_matrix.size()
        adjacency_matrix = adjacency_matrix.view(batch_size, num_nodes, num_nodes)

        X = self.linear(X.view(-1, X.size(2))).view(batch_size, num_nodes, -1)
        num_features = X.size(2)
        topK = self.topK
        X_topK, adjacency_matrix = get_topK_neighbor_nodes(
            X, adjacency_matrix, topK)
        #***********************************************************************
        #****************** COMPUTE PAIRWISE MESSAGES **************************
        X_pair_msg = X.view(batch_size, num_nodes, 1, num_features) + X_topK
        X_pair_msg = self.activation_function(X_pair_msg.view(-1, num_features))
        X_pair_msg = X_pair_msg.view(batch_size * num_nodes, topK, -1)
        #***********************************************************************
        #***************** AGGREGATE PAIRWISE MESSAGES *************************
        adjacency_matrix = adjacency_matrix.view(batch_size * num_nodes, 1, topK)
        X_message = torch.bmm(adjacency_matrix, X_pair_msg).unsqueeze(dim=1)
        X_message = X_message.view(batch_size, num_nodes, self.num_features_msg)
        #***********************************************************************

        return X_message


class RelationNetBasedGNN(nn.Module):
    def __init__(
        self,
        num_graph_layers,
        num_features_input,
        num_features_output,
        num_features_hidden,
        num_features_msg,
        aggregation_dropout,
        update_dropout,
        topK_neighbors):
        super(RelationNetBasedGNN, self).__init__()

        self.num_graph_layers = num_graph_layers

        if not isinstance(num_features_hidden, (list, tuple)):
            num_features_hidden = [num_features_hidden] * (num_graph_layers-1)
        assert(len(num_features_hidden) == num_graph_layers-1)

        if not isinstance(num_features_msg, (list, tuple)):
            num_features_msg = [num_features_msg] * num_graph_layers
        assert(len(num_features_msg) == num_graph_layers)

        aggregation_functions = []
        update_functions = []
        num_features_input_this = num_features_input
        for i in range(num_graph_layers):
            aggregation_function_this = RelationNetBasedAggregationFunction(
                num_features_in=num_features_input_this,
                num_features_msg=num_features_msg[i],
                dropout=aggregation_dropout,
                topK=topK_neighbors)

            if i == (num_graph_layers-1):
                update_function_this = UpdateFunctionFinal(
                    num_features_in=num_features_input_this,
                    num_features_msg=num_features_msg[i],
                    num_features=num_features_output)
            else:
                update_function_this = UpdateFunction(
                    num_features_in=num_features_input_this,
                    num_features_msg=num_features_msg[i],
                    num_features=num_features_hidden[i], dropout=update_dropout)

            aggregation_functions.append(aggregation_function_this)
            update_functions.append(update_function_this)
            num_features_input_this = update_function_this.num_features_out

        self.aggregation_functions = nn.ModuleList(aggregation_functions)
        self.update_functions = nn.ModuleList(update_functions)

    def forward(self, X_input, adjacency_matrix):
        for i in range(self.num_graph_layers):
            X_message = self.aggregation_functions[i](X_input, adjacency_matrix)
            X_output = self.update_functions[i](X_input, X_message)
            X_input = X_output

        return X_output


class MLP_OnlyUpdateFunctions(nn.Module):
    def __init__(
        self,
        num_layers,
        num_features_input,
        num_features_output,
        num_features_hidden,
        update_dropout):
        super(MLP_OnlyUpdateFunctions, self).__init__()

        self.num_layers = num_layers

        if not isinstance(num_features_hidden, (list, tuple)):
            num_features_hidden = [num_features_hidden] * (num_layers-1)
        assert(len(num_features_hidden) == num_layers-1)

        update_functions = []
        num_features_input_this = num_features_input
        for i in range(num_layers):
            if i == (num_layers-1):
                update_function_this = UpdateFunctionFinal(
                    num_features_in=num_features_input_this,
                    num_features_msg=0,
                    num_features=num_features_output)
            else:
                update_function_this = UpdateFunction(
                    num_features_in=num_features_input_this,
                    num_features_msg=0,
                    num_features=num_features_hidden[i], dropout=update_dropout)

            update_functions.append(update_function_this)
            num_features_input_this = update_function_this.num_features_out

        self.update_functions = nn.ModuleList(update_functions)

    def forward(self, X_input, adjacency_matrix):
        for i in range(self.num_layers):
            X_output = self.update_functions[i](X_input, None)
            X_input = X_output

        return X_output


class AdjacencyMatrixGenerator(nn.Module):
    """ Computes the adjancy matrix. """
    def __init__(self, temperature=10.0, learn_temperature=False):
        super(AdjacencyMatrixGenerator, self).__init__()
        self.temperature = nn.Parameter(
            torch.FloatTensor(1).fill_(temperature),
            requires_grad=learn_temperature)


    def forward(self, X_input, AdjacencyMatrixIdentity, SelfMaskMatrix=None):
        batch_size, num_nodes, num_features = X_input.size()
        X_input = F.normalize(X_input, p=2, dim=X_input.dim()-1, eps=1e-12)

        # AdjacencyMatrix shape: [batch_size, num_nodes, num_nodes]
        AdjacencyMatrix = torch.bmm(X_input, X_input.transpose(1, 2))

        mask = AdjacencyMatrixIdentity
        mask = mask.view(batch_size, num_nodes, num_nodes)

        AdjacencyMatrix = self.temperature * AdjacencyMatrix
        AdjacencyMatrix = AdjacencyMatrix -1e8 * mask
        AdjacencyMatrix = F.softmax(AdjacencyMatrix, dim=2)

        AdjacencyMatrix = AdjacencyMatrix.view(
            batch_size, num_nodes, num_nodes, -1)

        return AdjacencyMatrix


def residual_prediction(inputs, outputs):
    assert(outputs.dim()==3 and inputs.dim()==3)
    assert(inputs.size(0)==outputs.size(0))
    assert(inputs.size(1)==outputs.size(1))
    assert(outputs.size(2) == 2 * inputs.size(2))

    residuals, gates = torch.split(outputs, inputs.size(2), dim=2)
    residuals = F.normalize(residuals, p=2, dim=2, eps=1e-12)
    gates = torch.sigmoid(gates)
    outputs = inputs + gates * residuals

    return outputs


class WeightsDAE(nn.Module):
    def __init__(self, opt):
        super(WeightsDAE, self).__init__()

        self.dae_type = opt['dae_type']
        self.gaussian_noise = opt['gaussian_noise']
        self.step_size = opt['step_size'] if ('step_size' in opt) else 1.0

        if self.dae_type == 'RelationNetBasedGNN':
            self.Agenerator = AdjacencyMatrixGenerator(
                temperature=opt['temperature'],
                learn_temperature=opt['learn_temperature'])

            self.dae_network = RelationNetBasedGNN(
                num_graph_layers=opt['num_layers'],
                num_features_input=opt['num_features_input'],
                num_features_output=opt['num_features_output'],
                num_features_hidden=opt['num_features_hidden'],
                num_features_msg=opt['nun_features_msg'],
                aggregation_dropout=opt['aggregation_dropout'],
                update_dropout=opt['update_dropout'],
                topK_neighbors=opt['topK_neighbors'])
        elif self.dae_type == 'MLP_OnlyUpdateFunctions':
            self.dae_network = MLP_OnlyUpdateFunctions(
                num_layers=opt['num_layers'],
                num_features_input=opt['num_features_input'],
                num_features_output=opt['num_features_output'],
                num_features_hidden=opt['num_features_hidden'],
                update_dropout=opt['update_dropout'])

    def prepape_adjacency_matrix(self, input_weights):

        input_weights = input_weights.detach()
        batch_size, num_nodes = input_weights.size(0), input_weights.size(1)
        device = 'cuda' if  input_weights.is_cuda else 'cpu'
        adjacency_matrix_identity = torch.eye(
            num_nodes, device=device, requires_grad=False).view(
                1, num_nodes, num_nodes).repeat(batch_size, 1, 1)

        return self.Agenerator(input_weights, adjacency_matrix_identity)

    def prepare_dae_inputs(self, input_weights):
        if self.dae_type != 'MLP_OnlyUpdateFunctions':
            adjacency_matrix = self.prepape_adjacency_matrix(input_weights)
        else:
            adjacency_matrix = None

        if self.gaussian_noise > 0.0 and self.training:
            # Inject gaussian noise into the initial estimates of the
            # classification weights of the novel and base classes (i.e.,
            # weight_novel and weight_base tensors)
            device = 'cuda' if input_weights.is_cuda else 'cpu'
            input_weights_noise_vector = self.gaussian_noise * torch.randn(
                input_weights.size(), requires_grad=False, device=device)
            input_weights = input_weights + input_weights_noise_vector
            input_weights = F.normalize(
                input_weights, p=2, dim=input_weights.dim()-1, eps=1e-12)

        return input_weights, adjacency_matrix

    def forward(self, input_weights):
        input_weights, adjacency_matrix = self.prepare_dae_inputs(input_weights)

        output_weights = self.dae_network(input_weights, adjacency_matrix)
        output_weights = residual_prediction(input_weights, output_weights)

        if (not self.training) and self.step_size != 1.0:
            output_weights = F.normalize(
                output_weights, p=2, dim=output_weights.dim()-1, eps=1e-12)
            output_weights = (
                input_weights +
                self.step_size * (output_weights - input_weights))

        return output_weights
