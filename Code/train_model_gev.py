def train_model_gev(model_name=None, lambda_=None, lambda_2=None, config=None, checkpoint_dir=None, tuning=False,
                    validation=True, final_train=False, X_train=X_train_max, y_train=y_train_max):
    if tuning:
        if model_name == "LSTM_max":
            model = LSTM_max(n_features, sequence_len, batch_size, config["n_hidden"], config["n_layers"])
        # todo: tune for other models
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        if model_name == "M1_GEV":
            model = M1_GEV(input_size, batch_size, n_hidden)
        elif model_name == "FCN":
            model = FCN(sequence_len, batch_size, n_hidden)
        elif model_name == "M2_GEV":
            model = M2_GEV(n_features, sequence_len, batch_size, n_hidden, n_layers)
        elif model_name == "LSTM_GEV":
            model = LSTM_GEV(n_features, sequence_len, batch_size, n_hidden, n_layers)
        elif model_name == "DeepPIPE":
            model = LSTM_GEV(n_features, sequence_len, batch_size, n_hidden, n_layers)
        elif model_name == "M3_GEV":
            model = M3_GEV(n_features, sequence_len, batch_size, n_hidden, n_layers)
        elif model_name == "Trans":
            model = TransAm(feature_size=64, num_layers=2, dropout=0.0)
            # model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, worker_init_fn=seed_worker)
    if validation:
        validation_loader = DataLoader(TensorDataset(X_val_max, y_val_max), batch_size=batch_size,
                                       worker_init_fn=seed_worker)
        test_loader = DataLoader(TensorDataset(X_test_max, y_test_max), batch_size=batch_size,
                                 worker_init_fn=seed_worker)

    small_value = torch.tensor(0.05)
    zero_tensor = torch.tensor(0.0)
    q1 = torch.tensor(0.05)
    q2 = torch.tensor(0.95)
    y_max = y_train.max()
    y_min = y_train.min()

    mu_hat_all = torch.empty(0).to(device)
    sigma_hat_all = torch.empty(0).to(device)
    xi_hat_all = torch.empty(0).to(device)
    y_all = torch.empty(0).to(device)
    y_hat_all = torch.empty(0).to(device)
    y_q1_all = torch.empty(0).to(device)
    y_q2_all = torch.empty(0).to(device)

    xi_scipy, mu_init, sigma_init = torch.tensor(genextreme.fit(y_train.cpu()))
    xi_p_init = -xi_scipy + 0.05
    xi_n_init = -xi_scipy - 0.05

    # xi_p_init = torch.tensor(-0.5)
    # xi_n_init = torch.tensor(-0.6)

    print(f'y_max, y_min: {y_max, y_min}')
    # mu_init, sigma_init, xi_p_init, xi_n_init = zero_tensor, zero_tensor, zero_tensor, zero_tensor

    for epoch in (range(num_epochs)):
        with torch.autograd.set_detect_anomaly(True):
            for i, (inputs, labels) in enumerate(train_loader):
                if epoch == 0 and i == 0 and lambda_ > 0.0:
                    with torch.no_grad():
                        # mu_temp, sigma_temp, xi_p_temp, xi_n_temp, yhat_temp = model(inputs, y_max, y_min, zero_tensor, zero_tensor, zero_tensor, zero_tensor)
                        # print(f'initial values: mu (mean): {mu_temp.mean().item()}, sigma (mean): {sigma_temp.mean().item()}, xi_p (mean): {xi_p_temp.mean().item()}, xi_n (mean): {xi_n_temp.mean().item()}')
                        # mu_fix = mu_temp - mu_init
                        # sigma_fix = sigma_temp - sigma_init
                        # mu_temp, sigma_temp, xi_p_temp, xi_n_temp, yhat_temp = model(inputs, y_max, y_min, mu_fix, sigma_fix, zero_tensor, zero_tensor)
                        # xi_p_fix = xi_p_temp - xi_p_init
                        # xi_n_fix = xi_n_temp - xi_n_init
                        mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor

                if lambda_ > 0.0:
                    mu, sigma, xi_p, xi_n, yhat = model(inputs, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix)
                    # y_med = mu + (sigma/xi_p)*(((math.log(2.0))**(-xi_p)) - 1)
                    # y_q1 = mu + (sigma/xi_p)*(((-math.log(q1))**(-xi_p)) - 1)
                    # y_q2 = mu + (sigma/xi_p)*(((-math.log(q2))**(-xi_p)) - 1)
                    if epoch == 0 and i == 0:
                        print(
                            f'initial values after fixing:  mu (mean): {mu.mean().item()}, sigma (mean): {sigma.mean().item()}, xi_p (mean): {xi_p.mean().item()}, xi_n (mean): {xi_n.mean().item()}')
                        # break
                else:
                    mu, sigma, xi_p, xi_n, yhat = model(inputs, y_max, y_min, zero_tensor, zero_tensor, zero_tensor,
                                                        zero_tensor)

                if lambda_ > 0.0:
                    constraint = 1 + (xi_p / sigma) * (labels - mu)
                    count_constraint_violation.append(constraint[constraint < small_value].shape[0])
                    gev_loss = calculate_nll(labels.cpu(), mu.cpu(), sigma.cpu(), xi_p.cpu(), is_return=True) / (
                    labels.shape[0])
                    xi_rmse_loss = ((xi_p - xi_n) ** 2).mean().sqrt()
                    evt_loss = lambda_2 * gev_loss + (1 - lambda_2) * xi_rmse_loss
                rmse_loss = ((labels - yhat) ** 2).mean().sqrt()
                # print(labels.shape, yhat.shape)
                if lambda_ == 0.0:
                    train_loss = rmse_loss
                else:
                    train_loss = lambda_ * evt_loss + (1 - lambda_) * rmse_loss
                # print(f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(),4)} | EVT(NLL+RMSE(xi)): {round(evt_loss.item(),4)} | RMSE(y): {round(rmse_loss.item(),4)} | GEV(NLL): {round(gev_loss.item(),4)} | RMSE(xi_p_n): {round(xi_rmse_loss.item(),4)}| mu  sigma  xi_p xi_n: {round(mu.mean().item(), 4), round(sigma.mean().item(),4), round(xi_p.mean().item(),4),round(xi_n.mean().item(),4)}')

                if torch.isinf(train_loss.mean()) or torch.isnan(train_loss.mean()):
                    print("Constraint:\n", constraint, "GEV Loss:\n", gev_loss)
                    print("xi_p \n", xi_p, "ytruth \n", labels, "yhat \n", yhat)
                    # break
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            train_history[epoch] = train_loss.item()
            if lambda_ > 0.0:
                print(
                    f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(), 4)} | EVT(NLL+RMSE(xi)): {round(evt_loss.item(), 4)} | RMSE(y): {round(rmse_loss.item(), 4)} | GEV(NLL): {round(gev_loss.item(), 4)} | RMSE(xi_p_n): {round(xi_rmse_loss.item(), 4)}| mu  sigma  xi_p xi_n: {round(mu.mean().item(), 4), round(sigma.mean().item(), 4), round(xi_p.mean().item(), 4), round(xi_n.mean().item(), 4)} | constraint: {constraint.mean().item()}')
            else:
                print(f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(), 4)}')
            if validation:
                for j, (inputs, labels) in enumerate(validation_loader):
                    with torch.no_grad():
                        if lambda_ > 0.0:
                            mu, sigma, xi_p, xi_n, y_validation_predict = model(inputs, y_max, y_min, mu_fix, sigma_fix,
                                                                                xi_p_fix, xi_n_fix)
                        else:
                            mu, sigma, xi_p, xi_n, y_validation_predict = model(inputs, y_max, y_min, zero_tensor,
                                                                                zero_tensor, zero_tensor, zero_tensor)
                        rmse_loss = ((y_validation_predict - labels) ** 2).mean().sqrt()
                        validation_loss = rmse_loss
                validation_history[epoch] = validation_loss.item()

                for k, (inputs, labels) in enumerate(test_loader):
                    with torch.no_grad():
                        if lambda_ > 0.0:
                            mu, sigma, xi_p, xi_n, y_test_predict = model(inputs, y_max, y_min, mu_fix, sigma_fix,
                                                                          xi_p_fix, xi_n_fix)
                        else:
                            mu, sigma, xi_p, xi_n, y_test_predict = model(inputs, y_max, y_min, zero_tensor,
                                                                          zero_tensor, zero_tensor, zero_tensor)
                        rmse_loss = ((y_test_predict - labels) ** 2).mean().sqrt()
                        test_loss = rmse_loss
                        if (epoch == num_epochs - 1):
                            if lambda_ > 0.0:
                                # y_med = mu + (sigma/xi_p)*(((math.log(2.0))**(-xi_p)) - 1)
                                y_q1 = mu + (sigma / xi_p) * (((-math.log(q1)) ** (-xi_p)) - 1)
                                y_q2 = mu + (sigma / xi_p) * (((-math.log(q2)) ** (-xi_p)) - 1)

                                mu_hat_all = torch.cat((mu_hat_all, mu), 0)
                                sigma_hat_all = torch.cat((sigma_hat_all, sigma), 0)
                                xi_hat_all = torch.cat((xi_hat_all, xi_p), 0)
                                y_q1_all = torch.cat((y_q1_all, y_q1), 0)
                                y_q2_all = torch.cat((y_q2_all, y_q2), 0)
                            y_all = torch.cat((y_all, labels), 0)
                            y_hat_all = torch.cat((y_hat_all, y_test_predict), 0)
                test_history[epoch] = test_loss.item()

                # if (epoch % 2 - 1) == 0 and final_train == False:
                print(
                    f'Epoch {epoch}  | training loss: {train_loss.item()} | validation loss: {validation_loss.item()} | test loss: {test_loss.item()}')
                if tuning:
                    with tune.checkpoint_dir(epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save((model.state_dict(), optimizer.state_dict()), path)
            else:
                if epoch % 2 == 0 and final_train == False:
                    if lambda_ > 0.0:
                        print(
                            f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(), 4)} | EVT(NLL+RMSE(xi)): {round(evt_loss.item(), 4)} | RMSE(y): {round(rmse_loss.item(), 4)} | GEV(NLL): {round(gev_loss.item(), 4)} | RMSE(xi_p_n): {round(xi_rmse_loss.item(), 4)}| mu  sigma  xi_p xi_n: {round(mu.mean().item(), 4), round(sigma.mean().item(), 4), round(xi_p.mean().item(), 4), round(xi_n.mean().item(), 4)}')
                    else:
                        print(
                            f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(), 4)}  | RMSE(y): {round(rmse_loss.item(), 4)}')
                    if sum(count_constraint_violation) > 0: print(
                        f"Number of constraint violation: Total: {sum(count_constraint_violation)}")
            # print(f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(),4)} | EVT(NLL+RMSE(xi)): {round(evt_loss.item(),4)} | RMSE(y): {round(rmse_loss.item(),4)} | GEV(NLL): {round(gev_loss.item(),4)} | RMSE(xi_p_n): {round(xi_rmse_loss.item(),4)}| mu  sigma  xi_p xi_n: {round(mu.mean().item(), 4), round(sigma.mean().item(),4), round(xi_p.mean().item(),4),round(xi_n.mean().item(),4)}')
            if tuning: tune.report(validation_loss=(train_loss.item()), train_loss=train_loss.item())
    if not tuning: return model, mu_hat_all.detach(), sigma_hat_all.detach(), xi_hat_all.detach(), y_all.detach(), y_hat_all.detach(), y_q1_all.detach(), y_q2_all.detach()