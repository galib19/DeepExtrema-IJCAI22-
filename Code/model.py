class M3_GEV(nn.Module):

    def __init__(self, n_features, sequence_len, batch_size=64, n_hidden=10, n_layers=2):
        super(M3_GEV, self).__init__()

        self.n_hidden = n_hidden
        self.sequence_len = sequence_len
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0)
        self.fcn = nn.Linear(in_features=n_hidden * 2, out_features=4)
        self.fcn2 = nn.Linear(in_features=4, out_features=10)
        self.linear_y = nn.Linear(in_features=10, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden).to(device),
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden).to(device)
        )

    def forward(self, input_tensor, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix):
        self.reset_hidden_state()
        self.mu_fix = mu_fix
        self.sigma_fix = sigma_fix
        self.xi_p_fix = xi_p_fix
        self.xi_n_fix = xi_n_fix

        lstm_out, self.hidden = self.lstm(input_tensor.view(self.batch_size, self.sequence_len, -1),
                                          self.hidden)  # lstm_out (batch_size, seq_len, hidden_size*2)
        out = lstm_out[:, -1, :]  # getting only the last time step's hidden state of the last layer
        out = self.fcn(out)  # feeding lstm output to a fully connected network which outputs 3 nodes: mu, sigma, xi

        mu = out[:, 0] - self.mu_fix  # mu: first node of the fully connected network
        p1 = out[:, 1]  # sigma: second node of the fully connected network
        p2 = out[:, 2]
        p3 = out[:, 3]
        p2 = self.softplus(p2)
        p3 = self.softplus(p2)
        sigma = self.softplus(p1) - self.sigma_fix
        xi_p = ((sigma / (mu - y_min)) * (1 + boundary_tolerance) - (p2)) - self.xi_p_fix
        xi_n = ((p3) - (sigma / (y_max - mu)) * (1 + boundary_tolerance)) - self.xi_n_fix
        xi_p[xi_p > 0.95] = torch.tensor(0.95)

        out = self.fcn2(out)
        yhat = self.linear_y(out)

        return mu, sigma, xi_p, xi_n, yhat