import matplotlib.pyplot as plt
import seaborn as sns
import os


def plots(df, name):
    # Set the style of the plot
    sns.set_style("whitegrid")

    # Create the subplots with a horizontal space of 0.5 between them
    fig, axs = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={'wspace': 0.5})

    # Set the title of the figure and add some padding
    fig.suptitle(name, fontsize=26, y=1.1, fontweight='bold')

    # Set the title and axis labels of the subplots, and customize the lineplot
    axs[0].set_title("Epochs VS Train Loss", fontsize=18)
    axs[0].set_xlabel("Epoch", fontsize=14)
    axs[0].set_ylabel("Loss", fontsize=14)
    sns.lineplot(x='epoch', y='train_loss', data=df, ax=axs[0], color='red', linewidth=2)

    axs[1].set_title("Epochs VS Validation Loss", fontsize=18)
    axs[1].set_xlabel("epoch", fontsize=14)
    axs[1].set_ylabel("Loss", fontsize=14)
    sns.lineplot(x='epoch', y='val_loss', data=df, ax=axs[1], color='orange', linewidth=2)

    axs[2].set_title("Epochs VS Train Accuracy", fontsize=18)
    axs[2].set_xlabel("Epoch", fontsize=14)
    axs[2].set_ylabel("Train Accuracy", fontsize=14)
    sns.lineplot(x='epoch', y='train_acc', data=df, ax=axs[2], color='green', linewidth=2)

    axs[2].set_title("Epochs VS Val Accuracy", fontsize=18)
    axs[2].set_xlabel("Epoch", fontsize=14)
    axs[2].set_ylabel("Val Accuracy", fontsize=14)
    sns.lineplot(x='epoch', y='val_acc', data=df, ax=axs[2], color='green', linewidth=2)

    # plt.tight_layout(pad=2.5)
    fig_caption = f"Train Acc:{df.loc[df.shape[0]-1,'train_acc']}, Val Acc:{df.loc[df.shape[0]-1,'val_acc']}, Train loss:{df.loc[df.shape[0]-1,'train_loss']}, Val loss:{df.loc[df.shape[0]-1,'val_loss']}"
    fig.text(0.5, 0.1, fig_caption, ha='center', fontsize=14)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(os.getcwd(), "results/plots/"+name+".png"))
