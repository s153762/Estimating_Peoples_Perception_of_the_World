import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class AnalyseData:
    def __init__(self):
        sns.set_theme(style="ticks", font_scale=1.2)
        self.directory = "../../Test_data/Results_Test1_Test2/results_test1_test2.json"

    def main(self):
        data = self.get_data()
        _, test1, test2 = self.get_divided_keys_data(data)
        self.plot_test(data,test1)
        self.plot_test(data,test2,False)
        self.plot_test(data, test1+test2, False)

    def get_data(self):
        with open(self.directory) as f:
            data = json.load(f)
        return data

    def get_divided_keys_data(self, data):
        time_keyes = [k for k in data.keys() if "time" in k]
        test2_keyes = [k for k in data.keys() if "time" not in k and "Test2" in k]
        test1_keyes = [k for k in data.keys() if "time" not in k and "Test2" not in k]
        return time_keyes, test1_keyes, test2_keyes

    def get_data_from_keys(self, data,keys):
        filtered_data = [(k,self.convert(data[k]))for k in keys if k in data and "0-2" not in k]
        return filtered_data

    def convert(self, data):
        return [ p for f in data for p in f]

    def plot_test(self, data, keys, test1 = True):
        filtered_data = self.get_data_from_keys(data, keys)
        plt.figure(figsize=(10,6))
        for data_type in filtered_data:
            label = data_type[0]
            color = 'g' if "Glasses" in label else 'b' if "Emilie" in label else 'r' if "Sade" in label else "m"
            line = '-' if "310" in label else '--' if "240" in label else ':'
            data_type = data_type[1]
            plt.plot(range(len(data_type)),np.cumsum(data_type), label=label, color=color,linestyle=line)
        avg_len = int(np.round(sum([len(d[1]) for d in filtered_data]) / len(filtered_data)))
        if test1:
            plt.plot(range(avg_len),np.cumsum([0.75] * avg_len), color='y', label="Optimal", linestyle="-.")
            plt.title("Test 1", fontsize=20)
        else:
            plt.plot(range(avg_len), np.cumsum([0.01] * avg_len), color='y', label="Optimal", linestyle="-.")
            plt.title("Test 2", fontsize=20)
        plt.legend()
        plt.xlabel("Frames")
        plt.ylabel("Cumulative Probability")
        plt.show()

    def plot_test_average(self, data, keys):
        return ""



if __name__ == "__main__":
    AnalyseData().main()