class FCN(nn.Module):
  # output: max
  # train_loss: max
  def __init__(self, sequence_len, batch_size, n_hidden=50):
    super(FCN, self).__init__()

    self.batch_size = batch_size

    self.linear1 = nn.Linear(in_features=sequence_len, out_features=n_hidden)
    self.linear2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
    self.linear3 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
    self.linearFinal = nn.Linear(in_features=n_hidden, out_features=1)

  def forward(self, input_tensor, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix):

      out = self.linear1(input_tensor.view(self.batch_size, -1))
      out = torch.relu(out)
      out = self.linear2(out)
      out = torch.relu(out)
      out = self.linear3(out)
      out = torch.relu(out)
      y = self.linearFinal(out)

      mu = torch.tensor([0.0,0.0])
      sigma = torch.tensor([0.0,0.0])
      xi_p = torch.tensor([0.0,0.0])
      xi_n = torch.tensor([0.0,0.0])
      return mu, sigma, xi_p, xi_n, y


class LSTM_GEV(nn.Module):

    def __init__(self, n_features, sequence_len, batch_size=64, n_hidden=10, n_layers=2):
        super(LSTM_GEV, self).__init__()

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
        lstm_out, self.hidden = self.lstm(input_tensor.view(self.batch_size, self.sequence_len, -1),
                                          self.hidden)  # lstm_out (batch_size, seq_len, hidden_size*2)
        out = lstm_out[:, -1, :]  # getting only the last time step's hidden state of the last layer
        # print("hidden states mean, std, min, max: ", lstm_out[:,:,:].mean().item(), lstm_out[:,:,:].std().item(), lstm_out[:,:,:].min().item(), lstm_out[:,:,:].max().item()) # lstm_out.shape -> out.shape: 64,16,100 -> 64,16. Batch size: 64, input_seq_len:  16, n_hidden*2 = 50*2 = 100 // *2 for bidirectional lstm
        out = self.fcn(out)  # feeding lstm output to a fully connected network which outputs 3 nodes: mu, sigma, xi
        out = self.fcn2(out)
        y = self.linear_y(out)
        mu = torch.tensor([0.0, 0.0])
        sigma = torch.tensor([0.0, 0.0])
        xi_p = torch.tensor([0.0, 0.0])
        xi_n = torch.tensor([0.0, 0.0])
        return mu, sigma, xi_p, xi_n, y


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.feature_size = feature_size
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder1 = nn.Linear(train_time_steps * feature_size, 50)
        self.decoder2 = nn.Linear(50, 10)
        self.decoder3 = nn.Linear(10, 4)
        self.decoder4 = nn.Linear(4, 10)
        self.decoder5 = nn.Linear(10, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix):
        src = src.reshape(train_time_steps, batch_size, -1)
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = output.reshape(batch_size, train_time_steps * self.feature_size)
        output = self.decoder1(output)
        output = self.decoder2(output)
        output = self.decoder3(output)
        output = self.decoder4(output)
        output = self.decoder5(output)

        mu = torch.tensor([0.0, 0.0])
        sigma = torch.tensor([0.0, 0.0])
        xi_p = torch.tensor([0.0, 0.0])
        xi_n = torch.tensor([0.0, 0.0])
        return mu, sigma, xi_p, xi_n, output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class DeepPIPE(nn.Module):

    def __init__(self, n_features, sequence_len, batch_size=64, n_hidden=10, n_layers=2):
        super(DeepPIPE, self).__init__()

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
        self.fcn3 = nn.Linear(in_features=10, out_features=3)
        self.linear_y = nn.Linear(in_features=3, out_features=1)
        self.linear_p1 = nn.Linear(in_features=3, out_features=1)
        self.linear_p2 = nn.Linear(in_features=3, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden).to(device),
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden).to(device)
        )

    def forward(self, input_tensor, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix):
        self.reset_hidden_state()

        lstm_out, self.hidden = self.lstm(input_tensor.view(self.batch_size, self.sequence_len, -1),
                                          self.hidden)  # lstm_out (batch_size, seq_len, hidden_size*2)
        out = lstm_out[:, -1, :]  # getting only the last time step's hidden state of the last layer
        # print("hidden states mean, std, min, max: ", lstm_out[:,:,:].mean().item(), lstm_out[:,:,:].std().item(), lstm_out[:,:,:].min().item(), lstm_out[:,:,:].max().item()) # lstm_out.shape -> out.shape: 64,16,100 -> 64,16. Batch size: 64, input_seq_len:  16, n_hidden*2 = 50*2 = 100 // *2 for bidirectional lstm
        out = self.fcn(out)  # feeding lstm output to a fully connected network which outputs 3 nodes: mu, sigma, xi
        out = self.fcn2(out)
        out = self.fcn3(out)

        yhat = self.linear_y(out)
        p1 = self.linear_p1(out)
        p2 = self.linear_p2(out)
        p1 = self.softplus(p1)
        p2 = self.softplus(p2)

        mu = torch.tensor([0.0, 0.0])
        sigma = torch.tensor([0.0, 0.0])
        xi_p = torch.tensor([0.0, 0.0])
        xi_n = torch.tensor([0.0, 0.0])
        return mu, sigma, xi_p, xi_n, yhat, p1, p2

