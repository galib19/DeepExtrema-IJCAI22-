# @title
# plotting functions
def plot_losses(train_history, validation_history, test_history, ylim_low=0, ylim_high=0.05):
    plt.rcParams.update({'font.size': 20})
    plt.plot(train_history, label="Training loss")
    plt.plot(validation_history, label="Validation loss")
    plt.plot(test_history, label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim((ylim_low, ylim_high))
    plt.legend()
    plt.show()

#@title
def plot_scatter(x, y, model_name = None, is_download = False):
    rcParams['figure.figsize'] = 24, 10
    plt.style.use('default')
    plt.rcParams.update({'font.size': 20})
    ax = plt
    plt.scatter(x, y, color ='green', label= 'Predicted')
    plt.plot([x.min(), x.max()], [x.min(), x.max()], color='red', label = 'Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    # plt.suptitle(model_name)
    # function to show the plot
    if is_download:
        plt.savefig(model_name+'point.jpg')
        files.download(model_name+'point.jpg')
    plt.legend(loc='best')
    plt.show()


# @title
def plot_plot_y(y_t, y_p, y1=None, y2=None, model_name=None, is_download=False):
    rcParams['figure.figsize'] = 24, 10
    plt.style.use('default')
    plt.rcParams.update({'font.size': 22})
    x = torch.linspace(1, y_t.size()[0], y_t.size()[0])
    plt.plot(x, y_t, color='tomato', label='Actual')
    plt.plot(x, y_p, color='yellowgreen', label='Predictions')
    # if not (y1 is None):
    #   plt.plot(x, y1, color ='navajowhite', label = '10%Q')
    #   plt.plot(x, y2, color ='skyblue', label='90%Q')
    plt.fill_between(x, y1, y2, facecolor='lightgrey')
    envelop_patch = mpatches.Patch(color='lightgrey', label='Qunatile (90%)')
    plt.legend(handles=[envelop_patch])
    # plt.plot([0, x.max().item()], [0, x.max().item()], color='red', label = 'Actual')
    plt.xlabel('Datapoints')
    plt.ylabel('Block Maxima')
    # plt.suptitle(model_name)
    # function to show the plot
    plt.legend(loc='best')
    if is_download:
        plt.savefig(model_name + 'quantile.jpg')
        files.download(model_name + 'quantile.jpg')

    plt.show()

#@title
def plot_scatter_error(xi, error, model_name = None):
    # rcParams['figure.figsize'] = 16, 10
    # plt.style.use('fivethirtyeight')
    plt.scatter(xi, error, color ='green')
    plt.axhline(y=0.0, color='red')
    # plt.plot([0, 0], [0, 0])
    plt.xlabel('Actual')
    plt.ylabel('Error')
    plt.suptitle(model_name)
    # function to show the plot
    # plt.savefig(model_name+'.jpg')
    # files.download(model_name+'.jpg')
    plt.show()


def parameters_relation(y=None, z=None, parameter_name="xi", is_for_yhat=False, is_download=False):
    x = np.arange(0, y.shape[0], 1)
    if is_for_yhat:
        y = y.numpy()
    else:
        y = y.numpy()

    # rcParams['figure.figsize'] = 16, 10
    # plt.style.use('default')
    # plt.plot(y, z, color ='red')
    # plt.xlabel('Block Maxima')
    # plt.ylabel(str(parameter_name))
    # # function to show the plot
    # # plt.savefig(parameter_name+' line .jpg')
    # # files.download(parameter_name+' line.jpg')
    # plt.show()

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(z.min(), z.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(z)
    lc.set_linewidth(5)
    line = ax.add_collection(lc)
    ax.set_xlabel('Data-points')
    if is_for_yhat:
        ax.set_ylabel('Predicted Block Maxima')
    else:
        ax.set_ylabel('Actual Block Maxima')
    # ax.set_title('Relation with GEV Parameters')
    axcb = fig.colorbar(line, ax=ax)
    if parameter_name is 'xi':
        axcb.set_label(r'$\xi$')
    elif parameter_name is 'sigma':
        axcb.set_label(r'$\sigma$')
    elif parameter_name is 'mu':
        axcb.set_label(r'$\mu$')
    if is_download:
        fig.savefig(parameter_name + '.jpg')
        files.download(parameter_name + '.jpg')
    fig.show()

def all_result(y_all, yhat_all, y_q1_all, y_q2_all, mu_hat_all, sigma_hat_all, xi_hat_all, model_name="Test"):
    print("PICP: ", calculate_PICP(y_all,y_q1_all, y_q2_all))
    print("Correlation between actual and Predicted (mean): ", calculate_corr(y_all, yhat_all))
    print("RMSE of y (standardized): ", ((y_all - yhat_all) ** 2).mean().sqrt().item())
    y_all, yhat_all = inverse_scaler(y_all.tolist(), yhat_all.tolist())
    y_q1_all, y_q2_all = inverse_scaler(y_q1_all.reshape(-1,1).tolist(), y_q2_all.reshape(-1,1).tolist())
    print("RMSE of y : ", math.sqrt(mean_squared_error(y_all,yhat_all)))
    plot_scatter(y_all, yhat_all, model_name="Model M3: y estimations", is_download = False)
    y_all_sorted, indices = torch.sort(torch.from_numpy(y_all.flatten()))
    yhat_all_sorted = torch.from_numpy(yhat_all.flatten())[indices]
    y_q1_all_sorted = y_q1_all[indices]
    y_q2_all_sorted = y_q2_all[indices]
    xi_hat_all_sorted = xi_hat_all[indices]
    sigma_hat_all_sorted = sigma_hat_all[indices]
    mu_hat_all_sorted = mu_hat_all[indices]
    # print(yhat_all_sorted)
    # plt.plot(y_all_sorted, yhat_all_sorted)
    plot_plot_y(y_all_sorted.reshape(-1), yhat_all_sorted.reshape(-1), y1=y_q1_all_sorted.reshape(-1), y2= y_q2_all_sorted.reshape(-1), model_name="Model M3: y estimations", is_download = False)
    # parameters_relation(y=yhat_all_sorted, z=mu_hat_all_sorted, parameter_name="mu", is_for_yhat=True)
    # parameters_relation(y=yhat_all_sorted, z=sigma_hat_all_sorted, parameter_name="sigma", is_for_yhat=True)
    # parameters_relation(y=yhat_all_sorted, z=xi_hat_all_sorted, parameter_name="xi", is_for_yhat=True)
    parameters_relation(y=y_all_sorted.flatten(), z=mu_hat_all_sorted, parameter_name="mu", is_for_yhat=False, is_download=False)
    parameters_relation(y=y_all_sorted.flatten(), z=sigma_hat_all_sorted, parameter_name="sigma", is_for_yhat=False, is_download=False)
    parameters_relation(y=y_all_sorted.flatten(), z=xi_hat_all_sorted, parameter_name="xi", is_for_yhat=False, is_download=False)
    output = np.column_stack((y_all_sorted.flatten(),yhat_all_sorted.flatten(), y_q1_all_sorted.flatten(), y_q2_all_sorted.flatten(), xi_hat_all_sorted.numpy().flatten(), sigma_hat_all_sorted.numpy().flatten(), mu_hat_all_sorted.numpy().flatten()))
    np.savetxt(model_name+'.csv',output,delimiter=',')

def calculate_PICP(y_all,y_q1_all, y_q2_all):
    captured_data = 0
    total_data = y_all.shape[0]
    for i in range(total_data):
      if (y_all[i] < y_q2_all[i]) and y_all[i] > y_q1_all[i]: captured_data+=1
    PICP = captured_data/total_data
    return PICP
def calculate_corr(y_all,yhat_all):
    corr, _ = pearsonr(y_all.reshape(-1).cpu().numpy(), yhat_all.reshape(-1).cpu().numpy())
    return corr