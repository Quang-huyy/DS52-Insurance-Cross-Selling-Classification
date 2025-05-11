import matplotlib.pyplot as plt
import pandas as pd

class Visualization:
    def data_overview_visualization_1(self, df: pd.DataFrame, color):
        fig, axs = plt.subplots(4, 3, figsize=(16, 20))
        axs = axs.flatten()
        for index, ft in enumerate(df.columns):
            ax = axs[index]
            ax.hist(df[ft], bins=30, alpha=0.5, color=color)
            ax.set_title(ft)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Count")

        for j in range(len(df.columns), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()
    
    def data_overview_visualization_2(self, df: pd.DataFrame, color):
        fig, axs = plt.subplots(4, 3, figsize=(16, 20))
        axs = axs.flatten()
        for index, ft in enumerate(df.columns):
            ax = axs[index]
            counts = df[ft].value_counts()
            ax.bar(counts.index.astype(str), counts.values, alpha=0.5, color=color)
            ax.set_title(ft)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Count")
        for j in range(len(df.columns), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()
    
    def model_performance_plot(self, model_names, precision_list, recall_list):

        plt.figure(figsize=(8, 6))
        plt.scatter(precision_list, recall_list)

        for i, name in enumerate(model_names):
            plt.annotate(name, (precision_list[i], recall_list[i]), textcoords="offset points", xytext=(5,5), ha='left')
        
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision vs Recall per Model')
        plt.grid(True)
        plt.show()