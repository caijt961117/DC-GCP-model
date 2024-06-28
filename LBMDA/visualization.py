import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

def plot_comparison(num_files):
    for i in range(1, num_files + 1):
        
        filename1 = f'result_{i}.npy'
        filename2 = f'result_truth_{i}.npy'
        filename3 = f'result_no_truth_{i}.npy'

        data1 = np.load(filename1)
        data2 = np.load(filename2)
        data3 = np.load(filename3)

        # plt.figure(figsize=(15, 5))

        plt.subplot(3, 1, 1)
        plt.imshow(data1.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f'DA prediction: iteration_{i}')

        plt.subplot(3, 1, 2)
        plt.imshow(data2.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f'Ground truth: iteration_{i}')

        plt.subplot(3, 1, 3)
        plt.imshow(data3.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f'Results without DA: iteration_{i}')

        image_filename = f'comparison_{i}.png'
        plt.savefig(image_filename, dpi=600)
        plt.close()

        print(f'Comparison plot saved as {image_filename}')


# 假设我们有30个文件
num_files = 30
plot_comparison(num_files)
