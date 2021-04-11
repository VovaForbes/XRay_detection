import visdom
from datetime import datetime


class Visualizations:
    def __init__(self, env_name, thresholds):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None
        self.iou_mean = None
        self.iou = [None for _ in range(len(thresholds))]
        self.thresholds = thresholds

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss overall',
            )
        )

    def plot_iou(self, iou, step, idx):
        self.iou[idx] = self.vis.line(
            [iou],
            [step],
            win=self.iou[idx],
            update='append' if self.iou[idx] else None,
            opts=dict(
                xlabel='Step',
                ylabel='Precision',
                title='IOU={0:.2f} precision'.format(self.thresholds[idx])
            )
        )

    def plot_mean_iou(self, iou, step):
        self.iou_mean = self.vis.line(
            [iou],
            [step],
            win=self.iou_mean,
            update='append' if self.iou_mean else None,
            opts=dict(
                xlabel='Step',
                ylabel='Precision',
                title='Averaged precision over all thresholds'
            )
        )
