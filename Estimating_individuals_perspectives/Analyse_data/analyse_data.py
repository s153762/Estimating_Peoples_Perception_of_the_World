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
        self.plot_test(data,test1,"Test 1")
        plt.savefig("../result/test1.png")
        self.plot_test(data,test2,"Test 2")
        plt.savefig("../result/test2.png")
        self.plot_test(data, test1+test2, "Test 1 and Test 2")
        plt.savefig("../result/test1_test2.png")

        self.plot_test_average_both(data, test1, test2)
        plt.savefig("../result/average.png")

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
        filtered_data = [(k,self.flatten(data[k]))for k in keys if k in data and "0-2" not in k]
        return filtered_data

    def flatten(self, data):
        return [ p for f in data for p in f]

    def plot_test(self, data, keys, title):
        filtered_data = self.get_data_from_keys(data, keys)
        plt.figure(figsize=(10,6))
        for data_type in filtered_data:
            label = data_type[0]
            color = 'g' if "Glasses" in label else 'b' if "Emilie" in label else 'r' if "Sade" in label else "m"
            line = '-' if "310" in label else '--' if "240" in label else ':'
            data_type = data_type[1]
            plt.plot(range(len(data_type)),np.cumsum(data_type), label=label, color=color,linestyle=line)
        avg_len = int(np.round(sum([len(d[1]) for d in filtered_data]) / len(filtered_data)))
        if title == "Test 1":
            plt.plot(range(avg_len),np.cumsum([0.75] * avg_len), color='y', label="Optimal", linestyle="-.")
        elif title == "Test 2":
            plt.plot(range(avg_len), np.cumsum([0.01] * avg_len), color='y', label="Optimal", linestyle="-.")
        plt.title(title, fontsize=20)
        plt.legend()
        plt.xlabel("Frames")
        plt.ylabel("Cumulative Probability")
        plt.show()

    def plot_test_average(self, data, keys, test = True):
        filtered_data = self.get_average_data(data, keys)
        plt.figure(figsize=(10, 6))
        x = range(len(filtered_data))
        y = np.array(np.array(filtered_data)[:,1],dtype=np.float64)
        markerline, stemlines, baseline = plt.stem(x, y,markerfmt=' ')
        plt.setp(stemlines, 'linewidth', 10,'color','g')
        plt.xticks(x, np.array(filtered_data)[:,0])
        plt.plot(x, [sum(y)/len(y)]*len(x), label='Mean', linestyle='--')
        if test:
            plt.title("Test 1",fontsize=20)
        else:
            plt.title("Test 2",fontsize=20)
        plt.xlabel("Videos")
        plt.ylim([0,1])
        plt.ylabel("Average Probability")
        plt.show()

    def get_average_data(self, data, keys):
        filtered_data = [[k, self.average(data[k])] for k in keys if k in data and "0-2" not in k]
        return filtered_data

    def average(self, data):
        flatten = self.flatten(data)
        return sum(flatten)/len(flatten)

    def plot_test_average_both(self, data, keys1, keys2):
        plt.figure(figsize=(8, 6))

        # Test 1
        filtered_data1 = self.get_average_data(data, keys1)
        x = np.array(range(len(filtered_data1)-2))
        y = np.array(np.array(filtered_data1)[:-2,1],dtype=np.float64)
        markerline, stemlines, baseline = plt.stem(x-0.05, y,markerfmt=' ')
        plt.setp(stemlines, 'linewidth', 10, 'color', 'g',label="Test 1")
        plt.xticks(x, np.array(filtered_data1)[:-2,0])
        plt.plot(x, [sum(y)/len(y)]*len(x), label='Mean Test 1', linestyle='--')

        # Test 2
        filtered_data2 = self.get_average_data(data, keys2)
        x = np.array(range(len(filtered_data2)))+0.05
        y = np.array(np.array(filtered_data2)[:, 1], dtype=np.float64)
        markerline, stemlines, baseline = plt.stem(x, y, markerfmt=' ')
        plt.setp(stemlines, 'linewidth', 10, 'color', 'b',label="Test 2")
        plt.plot(x, [sum(y) / len(y)] * len(x), label='Mean Test 2', linestyle='--')

        plt.title("Test 1 and Test 2",fontsize=20)
        plt.xlabel("Videos")
        plt.ylim([0,0.7])
        plt.ylabel("Average Probability")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    AnalyseData().main()