import itertools
from sklearn.manifold import TSNE

def create_sequence_data(data, window_length = 2, flatten=False):
  x=data[:,:-1]
  y=data[:,-1]

  shape = (x.shape[0]-window_length+1, window_length, x.shape[1])
  strides = (x.strides[0], x.strides[0], x.strides[1])

  windowed_x = np.lib.stride_tricks.as_strided(x, shape, strides)
  matching_y = y[window_length-1:].reshape(-1,1)
  
  if flatten:
    windowed_data = np.append(windowed_x.reshape(windowed_x.shape[0],-1), matching_y, axis = 1)
  else:
    windowed_data = {'x':windowed_x, 'y':matching_y}
  
  return windowed_data

#cite: from Ilkay's gatech ml class
def save_or_show_plot(plt_name, save):
    if save:
        plt.savefig(plt_name)
        plt.close()
    else:
        plt.show()

#cite: from Ilkay's gatech ml class
def plot_histogram(data,name,save):
    f=data.hist(bins=int(data.shape[0]/100.0), figsize=(24, 24), xlabelsize=20, ylabelsize=20)
    plt.suptitle('Histograms for features of '+ name + ' dataset', fontsize=30, y=1.02)
    plt.tight_layout()
    [sp.title.set_size(30) for sp in f.flat]
    save_or_show_plot('hist'+name,save)

#cite: from Ilkay's gatech ml class
def get_corr_with_target(a, target):
    return pd.DataFrame((a.corrwith(target)).sort_values(ascending=False, key=abs), columns=['Correlation with target(' + target.name+')'])

#cite: from Ilkay's gatech ml class
def plot_table(data, name, type, save):
    f, ax = plt.subplots(1, 1, dpi=300)
    ax.axis('off')
    f.subplots_adjust(left=0.35, right=0.95)
    pd.plotting.table(ax,data, loc='center')
    plt.suptitle(type +' table for '+name+ ' data')
    plt.draw()
    save_or_show_plot(name+type, save)

#cite: from Ilkay's cs7641
def plot_2d_scatter_plots(data, columns,name, type, target, save):
    combs = list(itertools.combinations(columns, 2))
    dim = int(np.sqrt(len(combs)))+1
    f, axs = plt.subplots(dim, dim)
    f.tight_layout(rect=[0, 0, 1, 0.95])
    f.subplots_adjust(top=0.90, bottom=0.1, left=0.1)
    f.suptitle('Scatter plot between ' +type+ ' and target('+target + ') of '+ name +' dataset', fontsize=32)
    for i, ax in enumerate(f.axes):
        if i < len(combs):
            data.plot(ax=ax, kind="scatter", x=combs[i][0], y=combs[i][1], alpha=0.4, figsize=(20, 20), c=target, cmap=plt.get_cmap("jet"), colorbar=True)
            ax.set_ylabel(combs[i][0],fontsize = 24)
            ax.set_xlabel(combs[i][1],fontsize = 24)
    save_or_show_plot(name+'_scatter', save)

#cite: from Ilkay's cs7641
def plot_tsne(x_train, y_train, name, random_state, save):
    tsne = TSNE(random_state=random_state).fit_transform(x_train)
    plt.scatter(tsne[:,0],tsne[:,1],c=y_train.map({0:'orange', 1: 'blue'}), alpha=0.4)
    plt.title('t-SNE projection for features of ' + name + ' dataset')
    plt.figtext(0.15, 0.80, "Down Days", color='orange')
    plt.figtext(0.15, 0.85, "Up Days", color='blue')
    #plt.legend()
    save_or_show_plot('tsne'+name, save)
