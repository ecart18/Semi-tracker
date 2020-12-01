
import time
import matplotlib
import numpy as np
import cv2
from cell_tracker.normalizers import EqualizeHist, CLAHE, MinMax, RetinexMSRCP, RetinexMSRCR
from cell_tracker.segmenters import OtsuThresholding, BinaryThresholding, WaterShed, Unet

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

params = {
    'font.size': '8',
    'axes.titlesize': '8',
    'axes.labelsize': '8',
    'xtick.labelsize': '8',
    'ytick.labelsize': '8',
    'lines.linewidth': '1',
    'legend.fontsize': '7',
    'axes.labelpad': '2.0',
    'figure.figsize': '4, 4',
}
pylab.rcParams.update(params)


def plot_multi_lines(times, name):
    
    fig, ax = plt.subplots()

    x = np.array(times['size'])
    for key in times.keys():
        if key == 'size':
            continue
        # ax.plot(x, np.array(times[key]), label=key)
        ax.loglog(x, np.array(times[key]), label=key)
    ax.set_title(u'Running Times')
    ax.set_xlabel(u'Image Size (height/width)')
    ax.set_ylabel(u'Time/s')
    # ax.set_xlim(0, 10)
    # ax.set_ylim(-1.2, 1.2)
    # ax.set_xticks(np.arange(1, 11))
    # ax.set_xticks(np.arange(1, max(x)))
    ax.legend(loc=2)  # upper left corner
    # ax.minorticks_on()
    ax.tick_params(which='both', direction='in',
                bottom=True, top=True, left=True, right=True) 
    ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    fig.tight_layout()
    # plt.grid(True)
    plt.savefig(name, bbox_inches='tight', dpi=500)


def cal_normalizer_time(base_scale, base_img, ranges):
    process_methods = {
        'EqualizeHist': EqualizeHist(),
        'CLAHE': CLAHE(),
        'MinMax': MinMax(),
        'RetinexMSRCP': RetinexMSRCP(),
        'RetinexMSRCR': RetinexMSRCR(),
    }
    
    times = {}
    for process_method in process_methods.keys():
        times[process_method] = []
        process = process_methods[process_method]
        for scale_ratio in range(1, ranges):
            scale_img = cv2.resize(base_img, (scale_ratio*base_scale[0], scale_ratio*base_scale[1]))
            start = time.time()
            tmp = process(scale_img)
            end = time.time()
            times[process_method].append(end-start)
    times['size'] = [base_scale[0]*ratio for ratio in list(range(1, ranges))]
    return times
    
    
def cal_segmentor_time(base_scale, base_img, ranges):
    process_methods = {
        'OtsuThresholding': OtsuThresholding(),
        'BinaryThresholding': BinaryThresholding(threshold=129),
        'WaterShed': WaterShed(),
        # 'Unet': Unet(model_path='/Users/hutaobetter/Projects/cell-tracker/training_demo1/log/model_demo1.pth.tar')
    }
    
    times = {}
    for process_method in process_methods.keys():
        times[process_method] = []
        process = process_methods[process_method]
        for scale_ratio  in range(1, ranges):
            scale_img = cv2.resize(base_img, (scale_ratio*base_scale[0], scale_ratio*base_scale[1]))
            start = time.time()
            tmp = process(scale_img)
            end = time.time()
            times[process_method].append(end-start)
    times['size'] = [base_scale[0]*ratio for ratio in list(range(1, ranges))]
    return times



if __name__ == "__main__":
    
    img_path = '/Users/hutaobetter/Projects/cell-tracker/datasets/demo-dataset1/000000.jpg'
    base_scale = (64, 64)
    ranges = 3
    source_img = cv2.imread(img_path, -1)
    base_img = cv2.resize(source_img, base_scale)
    
    # norm_times = cal_normalizer_time(base_scale, base_img, ranges)
    # plot_multi_lines(norm_times, name='./norm.png')
    
    seg_times = cal_segmentor_time(base_scale, base_img, ranges)
    plot_multi_lines(seg_times, name='./seg.png')
    
    # import pdb; pdb.set_trace()
    
    
            
        
        
    
    