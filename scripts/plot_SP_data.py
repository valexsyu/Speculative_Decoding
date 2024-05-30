
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def corr_kl_accepted(files_path, save_figure_path, use_kl_shift=False):
    
    if len(files_path) != 2 :
        ValueError('The files path is error')
    else:
        kl_path = files_path[0]
        accepted_path = files_path[1]
        
    correlations_pearson = []
    correlations_spearman = []
    count_negative_pearson = 0
    count_positive_pearson = 0
    count_negative_spearman = 0
    count_positive_spearman = 0    

    with open(kl_path, 'r') as file1, open(accepted_path, 'r') as file2:
        
        for i, (line1, line2) in enumerate(zip(file1, file2)):
            data1 = np.fromstring(line1.strip(), sep=',')
            data2 = np.fromstring(line2.strip(), sep=',')

            # 处理数据长度不一致的情况
            if len(data1) != len(data2):
                print(f"Warning: 数据长度不一致 - 文件1长度为{len(data1)}, 文件2长度为{len(data2)}")
                min_length = min(len(data1), len(data2))
                data1 = data1[:min_length]
                data2 = data2[:min_length]

            # 替换无效值为10^10
            data1 = np.nan_to_num(data1, nan=1e10, posinf=1e10, neginf=-1e10)
            data2 = np.nan_to_num(data2, nan=1e10, posinf=1e10, neginf=-1e10)
            
            if use_kl_shift :
                data1_shifted = data1[:-1]
                data2_shifted = data2[1:]   
            else:         
                data1_shifted = data1
                data2_shifted = data2    
                
                 
            correlation_pearson = pearsonr(data1_shifted, data2_shifted)[0]
            correlation_spearman = spearmanr(data1_shifted, data2_shifted)[0]

            correlations_pearson.append(correlation_pearson)
            correlations_spearman.append(correlation_spearman)
            
            if correlation_pearson < 0:
                count_negative_pearson += 1
            elif correlation_pearson > 0:
                count_positive_pearson += 1
            if correlation_spearman < 0:
                count_negative_spearman += 1
            elif correlation_spearman > 0:
                count_positive_spearman += 1                

    print("Number of Pearson correlation coefficients less than 0 / greater than 0:", count_negative_pearson,count_positive_pearson )
    print("Number of Spearman correlation coefficients less than 0 / greater than 0:", count_negative_spearman, count_positive_spearman )   
    # print("Pearson correlation coefficients:", correlations_pearson)
    # print("Spearman correlation coefficients:", correlations_spearman)    



    # 绘制相关系数图表
    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.35
    index = np.arange(len(correlations_pearson))

    bar1 = ax.bar(index, correlations_pearson, bar_width, label=f'Pearson Correlation {count_negative_pearson}/{count_positive_pearson}')
    bar2 = ax.bar(index + bar_width, correlations_spearman, bar_width, label=f'Spearman Correlation {count_negative_spearman}/{count_positive_spearman}', color='orange')

    if use_kl_shift  :
        ax.set_title('Correlation Analysis KL_shift')
    else:
        ax.set_title('Correlation Analysis')
    ax.set_xlabel('Question Index')
    ax.set_ylabel('Correlation')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(index, fontsize=6)  # Adjust font size here
    ax.legend()

    plt.tight_layout()

    # 保存图表为PNG文件
    plt.savefig(save_figure_path)






def get_csv_files(directory_path, file_names):
    csv_files = []
    for root, dirs, files in os.walk(directory_path):
        for file_name in file_names :
            if file_name in files:
                if file_name.endswith('.csv'):
                    csv_files.append(os.path.join(root, file_name))
    if not csv_files:
        raise ValueError(f"File '{file_name}' not found in directory '{directory_path}'")
    return csv_files



def heatmap(files_path, save_figure_path=None):
    plt.figure(figsize=(10, 5))  # Set the size of the plot
    for file_path in files_path:
        # Extract file name from file path
        file_name = os.path.basename(file_path)
        
        # Read the data
        data = pd.read_csv(file_path, header=None)
        data.ffill(axis=1, inplace=True)  # Forward fill missing values in place

        # Create a figure and an axis for the plot
        fig, ax = plt.subplots()

        # Create a heatmap with a different colormap
        cax = ax.matshow(data, interpolation='nearest', cmap='viridis')
        fig.colorbar(cax)

        # Generate labels that are multiples of 10
        x_labels = [str(int(x) * 10) for x in range(data.shape[1])]
        y_labels = [str(int(y) * 10) for y in range(data.shape[0])]

        # Set axis labels using the proper tick locations and smaller font size
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))        
    



        ax.set_xticklabels(x_labels, fontsize=8)  # Set font size to 8
        ax.set_yticklabels(y_labels, fontsize=8)  # Set font size to 8

        # Set labels
        ax.set_title('Heatmap')


        # Save the figure if save_figure_path is provided
        if save_figure_path:
            plt.savefig(save_figure_path)
        plt.close(fig)  # Close the figure to free memory    




def accepted_number(files_path, save_figure_path=None, data_number=0,use_kl_shift=False):
    plt.figure(figsize=(10, 5))  # Set the size of the plot
    if len(files_path) < 2 :
        accepted_path = files_path[0] 
        kl_path = None
    else:
        kl_path = files_path[0]
        accepted_path = files_path[1]    

    file_name = os.path.basename(accepted_path)
       
    want_plot_num = 0
    # Open the file and read each line
    with open(accepted_path, 'r') as file:
        for line_num, line in enumerate(file):
            if line_num ==  data_number :
            
                # Split the line into values
                values = list(map(int, line.strip().split(',')))
                
                
                if use_kl_shift :
                    values = values[1:]   
                else:         
                    values = values   
                                    
                # Create x-axis values (column indices)
                x = np.arange(len(values))
                max_accepted_value = max(values)
                # plt.bar(x, values, label=f'MT_Bench_Q {data_number}')
                plt.bar(x, values, label=f'Data from {file_name} Qustoin {data_number}')
            else:
                continue     
                
    # Process the scatter plot data from kl_path if provided
    if kl_path:
        with open(kl_path, 'r') as kl_file:
            for line_num, line in enumerate(kl_file):
                if line_num ==  data_number :
                    # Split the line into values
                    values = list(map(float, line.strip().split(',')))
                    if use_kl_shift :
                        values = values[:-1]   
                    else:         
                        values = values                       
                    max_kl_value = max(values)
                    normalized_kl_values = [value*max_accepted_value / max_kl_value for value in values]
                    # Create x-axis values (column indices)
                    x_scatter = np.arange(len(normalized_kl_values))
                    plt.scatter(x_scatter, normalized_kl_values, color='red', label='KL Data')     
                else:
                    continue                                                   
                
    # Add labels and title
    plt.xlabel('Step')
    plt.ylabel('Times')
    plt.title('Accepted Number')

    # Set y-axis ticks to integers only
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add legend
    plt.legend()



    # Save the figure if save_figure_path is provided
    if save_figure_path:
        plt.savefig(save_figure_path)
    else:
    # Save the plot as an image
        plt.savefig('Accept_Number.png')        




    
    

# def inference_time(files_path, save_figure_path=None):
#     plt.figure(figsize=(10, 5))  # Set the size of the plot
#     print(f"Compare {len(files_path)} data ")
#     for i, file_path in enumerate(files_path):
#         print(f"data{i} from:{file_path}")
#         # Extract file name from file path
#         file_name = os.path.basename(file_path)
        
#         # Read the data
#         data = pd.read_csv(file_path, header=None, names=['Inference Time'])
        
#         if i > 0 :
#             ratio = np.mean(data) / first_avg
#         else:
#             first_avg = np.mean(data)
#             ratio = 1
#         # Plot the data
#         plt.plot(data['Inference Time'], label=f'{file_name} avg:{round(np.mean(data),3)}, ratio:{round(ratio,4)}')
            
#     # Add labels and title
#     plt.title('Inference Time')
#     plt.xlabel('Question Index')
#     plt.ylabel('Inference Time (ms)')

#     # Add legend to the plot
#     plt.legend()

#     # Save the figure if save_figure_path is provided
#     if save_figure_path:
#         plt.savefig(save_figure_path)
        
def inference_time(files_path, save_figure_path=None):
    plt.figure(figsize=(10, 5))  # Set the size of the plot
    print(f"Compare {len(files_path)} data ")
    for i, file_path in enumerate(files_path):
        print(f"data{i} from:{file_path}")
        # Extract file name from file path
        file_name = os.path.basename(file_path)
        
        # Read the data
        data = pd.read_csv(file_path, header=None, names=['Inference Time'])
        
        if i > 0 :
            ratio = data.div(first_data.values)
        else:
            first_data = data
            ratio = data.div(data.values)
            
        average = ratio.mean().values[0]
        # Plot the data
        plt.plot(data['Inference Time'], label=f'{file_name} ,ratio:{round(average,4)}')
            
    # Add labels and title
    plt.title('Inference Time')
    plt.xlabel('Question Index')
    plt.ylabel('Inference Time (ms)')

    # Add legend to the plot
    plt.legend()

    # Save the figure if save_figure_path is provided
    if save_figure_path:
        plt.savefig(save_figure_path)
        



def accepted_rate(files_path, save_figure_path=None):
    plt.figure(figsize=(10, 5))  # Set the size of the plot
    print(f"Compare {len(files_path)} data ")
    for i, file_path in enumerate(files_path):
        print(f"data{i} from:{file_path}")
        # Extract file name from file path
        file_name = os.path.basename(file_path)
        
        # Read the data
        data = pd.read_csv(file_path, header=None, names=['accepted_rate'])
        
        if i > 0 :
            ratio = data.div(first_data.values)
        else:
            first_data = data
            ratio = data.div(data.values)
            
        average = ratio.mean().values[0]
        # Plot the data
        plt.plot(data['accepted_rate'], label=f'{file_name} ratio:{round(average,4)}')
            
    # Add labels and title
    plt.title('Accepted Rate')
    plt.xlabel('Question Index')
    plt.ylabel('%')

    # Add legend to the plot
    plt.legend()

    # Save the figure if save_figure_path is provided
    if save_figure_path:
        plt.savefig(save_figure_path)        
        
        



def entropy(files_path, save_figure_path=None):
    plt.figure(figsize=(10, 5))  # Set the size of the plot  
    for file_path in files_path:
      # Extract file name from file path
        file_name = os.path.basename(file_path)
        
        mean_data_list = []
        variance_data_list = []
        
        with open(file_path, 'r') as file:
            for line in file:   
                # Convert string values to numeric (float or int)
                numeric_values = [float(value) for value in line.strip().split(',')]
                # Compute the mean and variance of numeric values in the line
                mean_data = np.mean(numeric_values)
                variance_data = np.var(numeric_values)
                mean_data_list.append(mean_data)
                variance_data_list.append(variance_data)

        # # Plot the mean data
        # plt.plot(mean_data_list, label=f'{file_name} (Mean)')
        # Plot the variance data with error bars
        plt.errorbar(
            x=range(len(variance_data_list)),
            y=mean_data_list,
            yerr=np.sqrt(variance_data_list),  # Standard deviation as error bars
            fmt='o',  # Use circles to denote data points
            label=f'{file_name} (Mean ± SD)'
        )



        

    # Add labels and title
    plt.title('Entropy')
    plt.xlabel('Question Index')
    plt.ylabel('Entropy')

    # Add legend to the plot
    plt.legend()

    # Save the figure if save_figure_path is provided
    if save_figure_path:
        plt.savefig(save_figure_path)




        
    

def main(args):
    files_path = get_csv_files(args.directory_path, args.file_names)
    if args.save_figure_name:
        save_figure_path = os.path.join(args.directory_path, args.args.save_figure_name)
    else:
        save_figure_path = None

    if args.plot_mode == 'inference_time':
        filename = f'Inferent_Time_{args.suffix}.png'
        save_figure_path = os.path.join(args.directory_path, filename) if save_figure_path is None else save_figure_path
        inference_time(files_path, save_figure_path)
    elif args.plot_mode == 'accepted_number':
        filename = f'Accepted_Number_{args.suffix}.png'
        save_figure_path = os.path.join(args.directory_path, filename) if save_figure_path is None else save_figure_path
        accepted_number(files_path, save_figure_path, args.data_number,args.use_kl_shift)
    elif args.plot_mode == 'heatmap':
        filename = f'Heatmap_{args.suffix}.png'
        save_figure_path = os.path.join(args.directory_path, filename) if save_figure_path is None else save_figure_path
        heatmap(files_path, save_figure_path)        
    elif args.plot_mode == 'corr_kl_accepted':
        filename = f'Correlation_KLDiv_AcceptRate_{args.suffix}.png'
        save_figure_path = os.path.join(args.directory_path, filename) if save_figure_path is None else save_figure_path
        corr_kl_accepted(files_path, save_figure_path, args.use_kl_shift)  
    elif args.plot_mode == 'accepted_rate':
        filename = f'Accepted_Rate_{args.suffix}.png'
        save_figure_path = os.path.join(args.directory_path, filename) if save_figure_path is None else save_figure_path
        accepted_rate(files_path, save_figure_path)                    
    elif args.plot_mode == 'entropy':
        filename = f'Entropy_{args.suffix}.png'
        save_figure_path = os.path.join(args.directory_path, filename) if save_figure_path is None else save_figure_path
        entropy(files_path, save_figure_path)              
                

        
    else:
        raise ValueError('Please provide a valid plot mode: inference_time or accepted_number')
    
    print(f"====Done====\n The file save in :{save_figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot performance metrics from CSV files in a directory.')
    parser.add_argument('directory_path', help='Path to the directory containing CSV files')
    parser.add_argument('file_names', nargs='+', help='CSV file names')
    parser.add_argument('--plot_mode', choices=['inference_time', 'accepted_number','accepted_rate', 'heatmap', 'corr_kl_accepted','entropy'], default='inference_time', help='Plot mode: inference_time or accepted_number')
    parser.add_argument('--save_figure_name', help='Path to save the figure as an image')
    parser.add_argument('--data_number', type=int, default=0, help='Specify the number of the data to plot.')
    parser.add_argument('--use_kl_shift', action='store_true', default=False, help='Shift the KL values.')
    parser.add_argument('--suffix', type=str, default="", help='.')


    args = parser.parse_args()
    main(args)
