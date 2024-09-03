from matplotlib import pyplot as plt
import numpy as np
import random
import os
random.seed(45)
print("i am jatin")
num_coins = 100
def toss(num_trials):
    '''
    num_trials: number of trials to be performed.
    
    return a numpy array of size num_trials with each entry representing the number of heads found in each trial

    Use for loops to generate the numpy array and 'random.choice()' to simulate a coin toss
    
    NOTE: Do not use predefined functions to directly get the numpy array. 
    '''
    global num_coins
    results = []
    print("num_trials=",num_trials)
    for _ in range(num_trials):
        heads_count=0
        for _ in range(num_coins):
            toss_results=random.choice(['heads', 'tails'])
            if toss_results== 'heads':
                heads_count += 1
           
        results.append(heads_count)
        '''print(f"Trial {num_trials + 1}: Number of heads = {heads_count}")  '''     
    return np.array(results)
    
    ## Write your code here

    return results
    

def plot_hist(trial):
    '''
    trial: vector of values for a particular trial.

    plot the histogram for each trial.
    Use 'axs' from plt.subplots() function to create histograms. You can search about how to use it to plot histograms.

    Save the images in a folder named "histograms" in the current working directory.  
    '''
    '''fig, axs = plt.subplots(figsize =(10, 7), tight_layout=True)'''
    # Check if the 'histograms' directory exists; if not, create it
    if not os.path.exists('histograms'):
        os.makedirs('histograms')

    # Create a figure and a set of subplots with specified size and tight layout
    fig, axs = plt.subplots(figsize=(10, 7), tight_layout=True)

    # Plot a histogram of the 'trial' data with specified bins and edge color
    axs.hist(trial, bins=range(min(trial), max(trial) + 1), edgecolor='black')

    # Set the title for the histogram
    axs.set_title('Histogram of Number of Heads')

    # Set the x-axis label
    axs.set_xlabel('Coins')

    # Set the y-axis label
    axs.set_ylabel('Number of heads')
    
    text_str = f'Number of Trials: {num_trials}'
    axs.text(0.95, 0.95, text_str, transform=axs.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right', color='blue')
    # Save the histogram as a PNG file in the 'histograms' directory
    plt.savefig('histograms/histogram_{}.png'.format(len(trial)))

    # Close the figure to free up memory
    plt.close(fig)
    
    ## Write your code here

if __name__ == "__main__":
    num_trials_list = [10,100,1000,10000,100000]
    for num_trials in num_trials_list:
        heads_array = toss(num_trials)
        plot_hist(heads_array)
