import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sb
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
import statistics


from models.LSTM_CNN import LSTM_CNN


def accuracy(output, target):
    """Computes the precision for the specified values of k"""
    batch_size = target.shape[0]
    correct = output.eq(target).sum() * 1.0
    acc = correct / batch_size

    return acc


# cite: from hw2
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# cite: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


# cite: from hw4
def train(model, dataloader, optimizer, criterion, device, scheduler=None, grad_clip=0):
    model.train()
    losses = AverageMeter()
    acc = AverageMeter()
    total_loss = 0.
    # progress_bar = tqdm_notebook(dataloader, ascii=True)
    progress_bar = dataloader
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        out = model(data.float())
        optimizer.zero_grad()
        # Note: input data to the BCEWithLogitsLoss can be anything. The 'Examples:' in the link below can generate any value, positive or negative
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        loss = criterion(out, target.float())
        total_loss += loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # plot_grad_flow(model.named_parameters())
        optimizer.step()

        # progress_bar.set_description_str(
        #    "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

        # pred = torch.round(torch.sigmoid(out)).long()
        # out[out > 0] = 1
        # out[out <= 0] = 0
        # pred = out.long()

        # Added back the rounding and sigmoid in training, validation, and testing in Trainer.py
        # Reference: https://stackoverflow.com/questions/64002566/bcewithlogitsloss-trying-to-get-binary-output-for-predicted-label-as-a-tensor
        pred = torch.round(f.sigmoid(out)).long()
        batch_acc = classification_report(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), output_dict=True)[
            'weighted avg']['f1-score']
        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

    return total_loss, losses.avg, acc.avg


# cite: from hw4
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.
    losses = AverageMeter()
    acc = AverageMeter()
    with torch.no_grad():
        # Get the progress bar
        # progress_bar = tqdm_notebook(dataloader, ascii=True)
        progress_bar = dataloader
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            out = model(data.float())
            # Note: input data to the BCEWithLogitsLoss can be anything. The 'Examples:' in the link below can generate any value, positive or negative
            # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
            loss = criterion(out, target.float())
            total_loss += loss
            # progress_bar.set_description_str(
            #    "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

            # pred = torch.round(torch.sigmoid(out)).long()
            # out[out > 0] = 1
            # out[out <= 0] = 0
            # pred = out.long()

            # Added back the rounding and sigmoid in training, validation, and testing in Trainer.py
            # Reference: https://stackoverflow.com/questions/64002566/bcewithlogitsloss-trying-to-get-binary-output-for-predicted-label-as-a-tensor
            pred = torch.round(f.sigmoid(out)).long()
            batch_acc = \
                classification_report(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), output_dict=True)[
                    'weighted avg']['f1-score']
            losses.update(loss, out.shape[0])
            acc.update(batch_acc, out.shape[0])

    return total_loss, losses.avg, acc.avg



def run_epoch(model, epochs, train_loader, valid_loader, optimizer, criterion, device):
    atl, ata, avl, ava = [], [], [], []
    for epoch in range(epochs):
      train_loss, atl_temp, ata_temp = train(model, train_loader, optimizer, criterion, device)
      _, avl_temp, ava_temp = evaluate(model, valid_loader, criterion, device)
      print("Epoch %d: Training Loss: %.4f. Training Acc: %.4f. Validation Loss: %.4f. Validation Acc: %.4f." % (epoch + 1, atl_temp, ata_temp, avl_temp, ava_temp))
      atl.append(atl_temp.item())
      ata.append(ata_temp)
      avl.append(avl_temp.item())
      ava.append(ava_temp)
    return atl, ata, avl, ava

# note: hardcoded model
def run_learning_curve_experiment(epochs, modeltype, HYPERPARAMETERS, train_dataset, valid_dataset, device, max_div, repetitions, reduction='mean'):
  result=np.zeros((repetitions,max_div,4))
  for k in range(0,repetitions):
    for j in range(0, max_div):
        print(f'repetition({k}) frac:({j/max_div})')
        train_dataset2 = torch.utils.data.Subset(train_dataset, range(0, len(train_dataset)//(max_div-j)))
        valid_dataset2 = torch.utils.data.Subset(valid_dataset, range(0, len(valid_dataset)//(max_div-j)))
        train_loader = DataLoader(train_dataset2, batch_size=HYPERPARAMETERS.BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset2, batch_size=HYPERPARAMETERS.BATCH_SIZE, shuffle=True)

        MODEL = LSTM_CNN(modeltype=modeltype, input_size=HYPERPARAMETERS.FEATURES, lstm_hidden_size=15, lstm_layers=6, lstm_output_size=1, kernel_size=3,padding=1, dropout=HYPERPARAMETERS.DROPOUT)

        CRITERION = torch.nn.BCEWithLogitsLoss(reduction='mean')
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=HYPERPARAMETERS.LR)
        MODEL.float()
        MODEL.to(device)
        atl, ata, avl, ava = run_epoch(MODEL, int(epochs*max_div/(j+1)), train_loader, valid_loader, OPTIMIZER, CRITERION, device)
        if reduction=='mean':
          result[k,j,:]=[statistics.mean(atl), statistics.mean(ata), statistics.mean(avl), statistics.mean(ava)]
        else:
          result[k,j,:]=[atl[-1], ata[-1], avl[-1], ava[-1]] #last element
  return (result.swapaxes(0,2))
  
# cite: from Ilkay's gatech ml class
def save_or_show_plot(plt_name, save):
    if save:
        plt.savefig(plt_name)
        plt.close()
    else:
        plt.show()


# cite: from Ilkay's gatech dl class
def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, info, save, path):
    plotting = {'Loss': {'tra_data': train_loss_history, 'val_data': valid_loss_history},
                'Accuracy': {'tra_data': train_acc_history, 'val_data': valid_acc_history}}
    for type, data in plotting.items():
        plt.plot(data['tra_data'], label='training')
        plt.plot(data['val_data'], label='validation')
        plt.title(type + " Curve " + info)
        plt.xlabel("Epoch")
        plt.ylabel(type)
        plt.grid()
        plt.legend(loc="best")
        save_or_show_plot(path + type + '_curve_' + info, save)


# cite: from Ilkay's gatech ml class
def plot_histogram(data, name, save):
    f = data.hist(bins=int(data.shape[0] / 100.0), figsize=(24, 24), xlabelsize=20, ylabelsize=20)
    plt.suptitle('Histograms for features of ' + name + ' dataset', fontsize=30, y=1.02)
    plt.tight_layout()
    [sp.title.set_size(30) for sp in f.flat]
    save_or_show_plot('hist' + name, save)


# cite: from Ilkay's gatech ml class
def get_corr_with_target(a, target):
    return pd.DataFrame((a.corrwith(target)).sort_values(ascending=False, key=abs),
                        columns=['Correlation with target(' + target.name + ')'])


# cite: from Ilkay's gatech ml class
def plot_table(data, name, type, save):
    f, ax = plt.subplots(1, 1, dpi=300)
    ax.axis('off')
    f.subplots_adjust(left=0.35, right=0.95)
    pd.plotting.table(ax, data, loc='center')
    plt.suptitle(type + ' table for ' + name + ' data')
    plt.draw()
    save_or_show_plot(name + type, save)


# cite: from Ilkay's cs7641
def plot_2d_scatter_plots(data, columns, name, type, target, save):
    combs = list(itertools.combinations(columns, 2))
    dim = int(np.sqrt(len(combs))) + 1
    f, axs = plt.subplots(dim, dim)
    f.tight_layout(rect=[0, 0, 1, 0.95])
    f.subplots_adjust(top=0.90, bottom=0.1, left=0.1)
    f.suptitle('Scatter plot between ' + type + ' and target(' + target + ') of ' + name + ' dataset', fontsize=32)
    for i, ax in enumerate(f.axes):
        if i < len(combs):
            data.plot(ax=ax, kind="scatter", x=combs[i][0], y=combs[i][1], alpha=0.4, figsize=(20, 20), c=target,
                      cmap=plt.get_cmap("jet"), colorbar=True)
            ax.set_ylabel(combs[i][0], fontsize=24)
            ax.set_xlabel(combs[i][1], fontsize=24)
    save_or_show_plot(name + '_scatter', save)


# cite: from Ilkay's cs7641
def plot_tsne(x_train, y_train, name, random_state, save):
    tsne = TSNE(random_state=random_state).fit_transform(x_train)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=y_train.map({0: 'orange', 1: 'blue'}), alpha=0.4)
    plt.title('t-SNE projection for features of ' + name + ' dataset')
    plt.figtext(0.15, 0.80, "Down Days", color='orange')
    plt.figtext(0.15, 0.85, "Up Days", color='blue')
    # plt.legend()
    save_or_show_plot('tsne' + name, save)

def plot_roc_auc(model, datasets, info, save, path):
    markers=['o','>','<','+','*','s','d','.']
    colors=['navy','orange', 'red','magenta']
    model.eval()
    for i,(type, dataset) in enumerate(datasets.items()):
        features, targets = dataset[:]
        predictions = f.sigmoid(model(features.float()))
        false_pos_rate, true_pos_rate, threshold = metrics.roc_curve(targets.cpu().detach().numpy(),   f.sigmoid(model(features.float())).cpu().detach().numpy())
        plt.plot(false_pos_rate, true_pos_rate, label = f'Area Under Curve ({type}) ={metrics.auc(false_pos_rate, true_pos_rate):.2f}', marker=markers[i%len(markers)], color=colors[i%len(colors)])
    plt.plot([0, 1], [0, 1],'orange')
    plt.title(f'Receiver operating characteristic ({info})')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend(loc = 'best')
    plt.grid()
    save_or_show_plot(path + 'roc_auc_curve_' + info, save)
    
# cite: from Ilkay's gatech dl class
def plot_learning_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, info, save, path):
    plotting = {'Loss': {'tra_data': train_loss_history, 'val_data': valid_loss_history},
                'Accuracy': {'tra_data': train_acc_history, 'val_data': valid_acc_history}}
    for type, data in plotting.items():
        plt.plot([i/len(data['tra_data']) for i in range(1, 1+len(data['tra_data']))],data['tra_data'], label='training')
        plt.plot([i/len(data['val_data']) for i in range(1, 1+len(data['val_data']))],data['val_data'], label='validation')
        plt.title(type + " Learning Curve " + info)
        plt.xlabel("Fraction of data used")
        plt.ylabel(type)
        plt.grid()
        plt.legend(loc="best")
        save_or_show_plot(path + type + '_learning_curve_' + info, save)
        
        
#cite: from Ilkay's gatech ml class
def get_pca_explained_variance_ratio(x_train, n_components, random_state):
    res={}
    temp = []
    for n_component in n_components:
        dim_reducer = PCA(n_components=n_component, random_state=random_state)
        reduced_x_train = dim_reducer.fit_transform(x_train)
        temp.append(sum(dim_reducer.explained_variance_ratio_))
    res['k'] = n_components
    res['total_explained_variance_ratio'] = temp
    return res

## EXAMPLE PCA RUN:
## assuming dataset already retrieved
#from sklearn.decomposition import PCA
#RANDOM_STATE=23817054
#base_sel=['SuperTrend1', 'SuperTrend2', 'SuperTrend3', 'BollingerUpperDifference', 'BollingerLowerDifference'] #pick columns for less clutter
#target_sel=['Next_Day_Change']
#dataset=dataset[base_sel+target_sel] #select a subset
#df_train = dataset[:][base_sel+target_sel]
#train_dataset = StockData(df_train.to_numpy(), num_days=HYPERPARAMETERS.NUM_DAYS)
#my_df=df_train
#x_temp,y_temp=train_dataset[:]
#x_temp,y_temp=x_temp.numpy(), y_temp.numpy()
#base_sel_expanded=[feature+ '-Day' + str(i//len(base_sel)) for i,feature in enumerate(HYPERPARAMETERS.NUM_DAYS*base_sel)]
#seq_data=np.append(x_temp.reshape(x_temp.shape[0],-1), y_temp, axis = 1)
#target_column='Day'+str(HYPERPARAMETERS.NUM_DAYS)+' Up/Down'
#seq_df=pd.DataFrame(data=seq_data,columns=base_sel_expanded+[target_column])
#seq_df_a=seq_df.sample(n=1000,random_state=RANDOM_STATE)#subsample for less clutter
#seq_df_a=seq_df_a.rename(columns={"BollingerUpperDifference-Day0": "Boll.Up.Dif-Day0", "BollingerUpperDifference-Day2": "Boll.Up.Dif-Day2"})
#x_train_only = seq_df_a.drop(target_column, axis=1)
#print(get_pca_explained_variance_ratio(x_train_only, list(range(1, len(base_sel_expanded))), RANDOM_STATE))
#pca = PCA(n_components=6, random_state=RANDOM_STATE)
#reduced_x_train = pca.fit_transform(x_train_only)
#pca_result = {'reduced_x':reduced_x_train, 'components':pca.components_}
#plot_name = 'SPX_PCA_biplot_loadings'
#chosen_component0=0
#chosen_component1=1
#plot_pca_k_val(pca_result, seq_df_a[target_column], chosen_component0, chosen_component1, x_train_only.columns, name=plot_name, save=False,load_thld=0.0)

#cite https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot                
def plot_pca_k_val(exp_results, labels, pc0, pc1,features, name, save,load_thld=0, show_loads=True):
    xs = exp_results['reduced_x'][:, pc0]
    ys = exp_results['reduced_x'][:, pc1]
    M = exp_results['components'].shape[0]
    #plt.figure(figsize=(int(M/3)+3, 3+int(len(features)/6)))
    #sb.heatmap(exp_results['components'].T, xticklabels=['C'+str(i) for i in range(0,M)], yticklabels=features,cmap=sb.color_palette("mako", as_cmap=True))
    #plt.title((name+'_loadings').replace("_", " "))
    #save_or_show_plot(name, save)

    coeff = exp_results['components'][[pc0,pc1]].T
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    scatter = plt.scatter(xs * scalex, ys * scaley, c=labels, alpha=0.8,cmap='Set3')
    plt.legend(handles=scatter.legend_elements()[0], labels=[0,1],title="Target-Label", loc='lower left')
    if show_loads:
        coeff=0.5 * coeff / np.abs(coeff).max()
        for i in range(n):
            if (coeff[i, 0]**2 + coeff[i, 1]**2)>(0.25 * np.abs(coeff).max())**2:
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.8)
                plt.text(coeff[i, 0] * .9, coeff[i, 1] * .9, features[i], color='k', ha='center', va='center')
    scale=0.6
    plt.xlim(-1*scale, scale)
    plt.ylim(-1*scale, scale)
    plt.xlabel("Principal C.{}".format(pc0))
    plt.ylabel("Principal C.{}".format(pc1))
    plt.title(name.replace("_", " "))
    plt.grid()
    save_or_show_plot(name, save)


## EXAMPLE T-SNE RUN:
## assuming dataset already retrieved
#base_sel=['SuperTrend1', 'SuperTrend2', 'SuperTrend3', 'BollingerUpperDifference', 'BollingerLowerDifference'] #pick columns for less clutter
#target_sel=['Next_Day_Change']
#dataset=dataset[base_sel+target_sel] #select a subset
#df_train = dataset[:][base_sel+target_sel]
#train_dataset = StockData(df_train.to_numpy(), num_days=HYPERPARAMETERS.NUM_DAYS)
#my_df=df_train
#x_temp,y_temp=train_dataset[:]
#x_temp,y_temp=x_temp.numpy(), y_temp.numpy()
#base_sel_expanded=[feature+ '-Day' + str(i//len(base_sel)) for i,feature in enumerate(HYPERPARAMETERS.NUM_DAYS*base_sel)]
#seq_data=np.append(x_temp.reshape(x_temp.shape[0],-1), y_temp, axis = 1)
#target_column='Day'+str(HYPERPARAMETERS.NUM_DAYS)+' Up/Down'
#seq_df=pd.DataFrame(data=seq_data,columns=base_sel_expanded+[target_column])
#plot_tsne(seq_df.drop(target_column, axis=1), seq_df[target_column],'SPX', 512, save=False)

#cite: from Ilkay's gatech ml class
def plot_tsne(x_train, y_train, name, random_state, save):
    tsne = TSNE(random_state=random_state).fit_transform(x_train)
    plt.scatter(tsne[:,0],tsne[:,1],c=y_train.map({0:'orange', 1: 'blue'}), alpha=0.4)
    plt.title('t-SNE projection for features of ' + name + ' dataset')
    plt.figtext(0.15, 0.80, "Down Days", color='orange')
    plt.figtext(0.15, 0.85, "Up Days", color='blue')
    #plt.legend()
    save_or_show_plot('tsne'+name, save)
