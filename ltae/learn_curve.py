import json
import matplotlib.pyplot as plt
import os


def plot_curve(data, title):
    """
    data: path to Json file
    title: title of the plot
    """
    data = json.load(open(data))
    epoch =range(1, len(data)+1)
    
    # Loss
    train_loss = [data[str(i)]['train_loss'] for i in epoch]
    val_loss = [data[str(i)]['val_loss'] for i in epoch]

    # Accuracy
    train_accuracy = [data[str(i)]['train_accuracy'] for i in epoch]
    val_accuracy = [data[str(i)]['val_accuracy'] for i in epoch]

    # IoU
    train_IoU = [data[str(i)]['train_IoU'] for i in epoch]
    val_IoU = [data[str(i)]['val_IoU'] for i in epoch]

 
    def get_max_IoU():
        max_val_IoU = max(val_IoU)
        max_val_epoch = val_IoU.index(max_val_IoU)
        return max_val_IoU, max_val_epoch
    max_val_IoU, max_val_epoch = get_max_IoU()
    # plot loss and accuracy in separate figure axis
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
    
    # Loss
    ax1.plot(epoch, train_loss, label='train loss')
    ax1.plot(epoch, val_loss, label='val loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy
    ax2.plot(epoch, train_accuracy, label='train accuracy')
    ax2.plot(epoch, val_accuracy, label='val accuracy')
    
    # plot max_train_epoch and max_val_epoch point
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    
    # IoU
    ax3.plot(epoch, train_IoU, label='train IoU')
    ax3.plot(epoch, val_IoU, label='val IoU')
    ax3.scatter(max_val_epoch, max_val_IoU, color = "#FF7C08")
    ax3.annotate('max: %.2f at %.f epoch' % (max_val_IoU, max_val_epoch), xy=(max_val_epoch, max_val_IoU), xytext=(max_val_epoch, max_val_IoU+0.01))
    ax3.set_title('Training and Validation IoU')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU')
    ax3.legend()
    
    fig.suptitle(title)
    # fig.tight_layout()
    # plt.savefig(os.path.join('./charts', title + '.png'),dpi=1500, transparent=False)
    plt.show()
    
# files = ["../../../results/ltae/results/Seed_0/trainlog.json", 
#          "../../../results/ltae/results/2018/Seed_0/seed_0_batch_128_epoch_10_lr_0.01_trainlog.json", 
#          "../../../results/ltae/results/2018/Seed_0/seed_0_batch_128_epoch_10_lr_0.001_sch_trainlog.json", 
#          "../../../results/ltae/results/2018/Seed_0/seed_0_batchsize_128_epochs_35_factor_84253_trainlog.json",
#          "../../../results/ltae/trials/Seed_0/trainlog.json",
#         "../../../results/ltae/results/2018/Seed_0/seed_0_batchsize_2048_epochs_100_fator_1300_trainlog.json",
#          "../../../results/ltae/results/2018/Seed_0/seed_0_batchsize_2048_epochs_100_factor_1000_trainlog.json",
#          "../../../results/ltae/results/2018/Seed_0/seed_0_batchsize_2048_epochs_30_factor_5266_sch_trainlog.json",
#          "../../../results/ltae/results/2018/Seed_0/seed_0_batchsize_2048_epochs_30_factor_5266_pos_trainlog.json",
#          "../../../results/ltae/results/2018/Seed_0/seed_0_weight_0.0001_batchsize_2048_epochs_30_factor_5266_trainlog.json"
#         ]
# category = ["Category-A", "Category-B", "Category-C", 
#             "Category-D",
#             "Category-E", "Category-F", "Category-G", "Category-H", "Category-H1", "Category-H2" ]
# len(files)
# for i in range(len(files)):
#     plot_curve(files[i], category[i])

# file_ = "../../../results/ltae/results/2018/Seed_0/seed_0_batchsize_2048_epochs_15_factor_5266_trainlog.json"
# plot_curve(file_, "")