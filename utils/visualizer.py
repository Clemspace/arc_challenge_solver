import matplotlib.pyplot as plt

class ARCVisualizer:
    @staticmethod
    def visualize_comparison(input_grid, expected_output, predicted_output, binary_loss, abs_diff_loss):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(input_grid, cmap='tab20')
        ax1.set_title('Input')
        ax2.imshow(expected_output, cmap='tab20')
        ax2.set_title('Expected Output')
        ax3.imshow(predicted_output, cmap='tab20')
        ax3.set_title(f'Predicted Output\nBinary Loss: {binary_loss}\nAbs Diff Loss: {abs_diff_loss:.2f}')
        plt.show()
