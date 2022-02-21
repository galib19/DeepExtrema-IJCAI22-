def calculate_nll(block_maxima, mu, sigma, xi, name="Test", is_return = False):
  size =  block_maxima.shape[0]
  block_maxima = torch.flatten(block_maxima.cpu())
  if not torch.is_tensor(mu):
      mu = torch.from_numpy(mu).float().to(device)
  if not torch.is_tensor(sigma):
      sigma = torch.from_numpy(sigma).float().to(device)
  if not torch.is_tensor(xi):
      xi = torch.from_numpy(xi).float().to(device)
  if mu.numel() == 1:
      mu = torch.flatten(torch.full((size,1), mu))
  if sigma.numel() == 1:
      sigma = torch.flatten(torch.full((size,1), sigma))
  if xi.numel() == 1:
      xi = torch.full((size,1), xi)
  mu = torch.flatten(mu).cpu()
  sigma = torch.flatten(sigma).cpu()
  xi = torch.flatten(xi).cpu()

  #using library
  log_pdf = genextreme.logpdf(block_maxima, loc = mu.detach().numpy(), scale = sigma.detach().numpy(), c = -xi.detach().numpy())
  log_likelihood = np.sum(log_pdf)
  #using vector
  # print(xi.shape, block_maxima.shape, mu.shape, sigma.shape)
  constraint = 1+(xi/sigma)*(block_maxima-mu)
  # constraint = constraint[constraint>0]
  constraint[constraint<0.05] = torch.tensor(0.5)
  first_term = torch.sum(torch.log(sigma))
  second_term =  (torch.sum((1+1/xi)*torch.log(constraint)))
  third_term =  torch.sum(constraint**(-1/xi))
  nll = (first_term + second_term + third_term)
  if is_return:
      return nll
  else:
      print("\n"+name+": \n")
      print("negative log likelihood using library:", -log_likelihood, " and using vector:", nll.item())
      print(f"first_term: {first_term}, second_term: {second_term}, third_term: {third_term}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def plot_histogram(data, plot_name="Test"):
  data = np.array(data.cpu())
  plt.figure(figsize=(10,5))
  plt.hist(data, bins = 55)
  title = "Histogram of " +plot_name
  plt.title(title)
  plt.show()

def inverse_scaler(predictions, actuals):
    predictions_inverse_scaler = scaler.inverse_transform(predictions)
    actuals_inverse_scaler = scaler.inverse_transform(actuals)
    return predictions_inverse_scaler, actuals_inverse_scaler


def extend_last_batch(X_h, X_m, X_mask, y, batch_size=batch_size):
    last_batch_size = X_h.shape[0] % batch_size
    X_mask = torch.tensor(X_mask).to(device)
    if last_batch_size != 0:
        if y.shape[0] - last_batch_size == 0:
            indices = [i for i in range(0, y.shape[0])]
        else:
            indices = [i for i in range(0, (y.shape[0] - last_batch_size))]
        # index = random.sample(indices, batch_size - last_batch_size)
        index = indices[-(batch_size - last_batch_size):]
        X_h_extended = X_h[index]
        X_m_extended = X_m[index]
        X_mask_extended = X_mask[index]
        y_extended = y[index]
        X_h = torch.cat((X_h, X_h_extended), 0)
        X_m = torch.cat((X_m, X_m_extended), 0)
        X_mask = torch.cat((X_mask, X_mask_extended), 0)
        y = torch.cat((y, y_extended), 0)
    return X_h, X_m, X_mask, y


def create_X_data(dataset, time_step=1):
    dataX = []
    for i in range(len(dataset)):
        X_data = dataset[i][0:time_step]
        dataX.append(X_data)
    return np.array(dataX)


def ready_X_data(train_data, val_data, test_data, train_time_steps):
    X_train = create_X_data(train_data, train_time_steps)
    X_val = create_X_data(val_data, train_time_steps)
    X_test = create_X_data(test_data, train_time_steps)
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    X_train = torch.from_numpy(X_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)

    return X_train, X_val, X_test


def create_y_data(dataset, time_step=1):
    dataY = []
    for i in range(len(dataset)):
        y_data = np.max(dataset[i][time_step:])
        dataY.append(y_data)
    return np.array(dataY)


def ready_y_data(train_data, val_data, test_data, train_time_steps):
    y_train = create_y_data(train_data, train_time_steps)
    y_val = create_y_data(val_data, train_time_steps)
    y_test = create_y_data(test_data, train_time_steps)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    return y_train, y_val, y_test


def ready_X_m_data(train_data, val_data, test_data, test_time_steps):
    tr, vl, ts = train_data.reshape(train_data.shape[0], test_time_steps, 21), val_data.reshape(val_data.shape[0],
                                                                                                test_time_steps,
                                                                                                21), test_data.reshape(
        test_data.shape[0], test_time_steps, 21)
    X_train_m = torch.from_numpy(tr).float().to(device)
    X_val_m = torch.from_numpy(vl).float().to(device)
    X_test_m = torch.from_numpy(ts).float().to(device)

    return X_train_m, X_val_m, X_test_m

def extend_last_batch_m(X_m, y, batch_size=batch_size):
    last_batch_size = X_m.shape[0] % batch_size
    if last_batch_size != 0:
        if y.shape[0]-last_batch_size ==0:
            indices = [i for i in range(0, y.shape[0])]
        else: indices = [i for i in range(0, (y.shape[0]-last_batch_size))]
        # index = random.sample(indices, batch_size - last_batch_size)
        index = indices[-(batch_size - last_batch_size):]
        X_m_extended = X_m[index]
        y_extended = y[index]
        X_m = torch.cat((X_m, X_m_extended), 0)
        y = torch.cat((y, y_extended), 0)
    return X_m, y

#@title
def create_gev_data(mu = 0.4, sigma=0.1, xi=0.25, size=y_train.shape[0]):
    data = genextreme.rvs(c=-xi, loc=mu, scale=sigma, size = size, random_state=RANDOM_SEED)
    block_maxima = data.tolist()
    shape, loc, scale = genextreme.fit(block_maxima)
    print(f"Ground Truth: mu: {mu}, sigma: {sigma}, xi: {xi}")
    print(f"Scipy Estimated GEV Parameters: mu: {loc}, sigma: {scale}, xi: {- shape}")
    # print("Lower Bound:", loc - scale/shape)
    y_truth =torch.from_numpy(data).float().to(device)
    X_dummy = torch.ones(size)
    # print("\ny's shape and X's shape:", y_truth.shape[0], " x ", X_dummy.shape[0])
    return X_dummy, y_truth 