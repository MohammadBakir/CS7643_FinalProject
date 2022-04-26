import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
import torch


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
        loss = criterion(out, target.float())
        total_loss += loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # plot_grad_flow(model.named_parameters())
        optimizer.step()

        # progress_bar.set_description_str(
        #    "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

        pred = torch.round(torch.sigmoid(out)).long()
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
            loss = criterion(out, target.float())
            total_loss += loss
            # progress_bar.set_description_str(
            #    "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

            pred = torch.round(torch.sigmoid(out)).long()
            batch_acc = \
            classification_report(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), output_dict=True)[
                'weighted avg']['f1-score']
            losses.update(loss, out.shape[0])
            acc.update(batch_acc, out.shape[0])

    return total_loss, losses.avg, acc.avg


# cite: from Ilkay's gatech ml class
def save_or_show_plot(plt_name, save):
    if save:
        plt.savefig(plt_name)
        plt.close()
    else:
        plt.show()


# cite: from Ilkay's gatech dl class
def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, info, save):
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
        save_or_show_plot(type + '_curve_' + info, save)


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
