import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class AnalyseData:
    def __init__(self):
        sns.set_theme(style="ticks", font_scale=1.2)
        self.directory = "../../Test_data/Resultat_test1_test2/Results_hele_klip/results_test1_test2.json"
        self.directory_divided = "../../Test_data/Resultat_test1_test2/opdelt_med_30/results_test1_test2.json"#"../../Test_data/Resultat_Opdelt_test1_test2/results_test1_test2.json"

    def main(self):
        #data = self.get_data()
        #_, test1, test2 = self.get_condition_keys_data(data)

        #self.plot_test(data,test1,"Test 1")
        #plt.savefig("../result/test1.png")
        #self.plot_test(data,test2,"Test 2")
        #plt.savefig("../result/test2.png")
        #self.plot_test(data, test1+test2, "Test 1 and Test 2")
        #plt.savefig("../result/test1_test2.png")
        #self.plot_average_both(data, test1, test2)
        #plt.savefig("../result/average.png")

        #self.plot_violinplot(data, test1, test2)
        #plt.savefig("../result/violin.png")

        data_divided = self.get_data_dir(self.directory_divided)
        _, keys1 = self.get_orientation_keys_data(data_divided, "test1")
        self.plot_violinplot_orientation(data_divided, keys1, [2,2], ["","Right","Front","Left"])
        plt.savefig("../result/violin_test1_30.png")

        _, keys2 = self.get_orientation_keys_data(data_divided, "test2")
        self.plot_violinplot_orientation(data_divided, keys2, [2,2],["Back","Right","Front","Left"])
        plt.savefig("../result/violin_test2_30.png")

        self.plot_violinplot_orientation(data_divided, [keys1[1], keys2[1]] , [2,2],["Test 1: Right","Test2: Right"])
        plt.savefig("../result/violin_right_30.png")
        self.plot_violinplot_orientation(data_divided, [keys1[2], keys2[2]], [2, 2], ["Test 1: Front", "Test2: Front"])
        plt.savefig("../result/violin_front_30.png")
        self.plot_violinplot_orientation(data_divided, [keys1[3], keys2[3]], [2, 2], ["Test 1: Left", "Test2: Left"])
        plt.savefig("../result/violin_left_30.png")
        plt.show()

        self.plot_violinplot_all(data_divided, [self.flatten(keys1), self.flatten(keys2)])
        plt.savefig("../result/violin_combined_30.png")

    def get_data(self):
        with open(self.directory) as f:
            data = json.load(f)
        return data

    @staticmethod
    def get_data_dir(dir):
        with open(dir) as f:
            data = json.load(f)
        return data

    def get_condition_keys_data(self, data):
        time_keyes = [k for k in data.keys() if "time" in k]
        test2_keyes = [k for k in data.keys() if "time" not in k and "Test2" in k]
        test1_keyes = [k for k in data.keys() if "time" not in k and "Test2" not in k]
        return time_keyes, test1_keyes, test2_keyes

    def get_orientation_keys_data(self, data, test):
        time_keyes = [k for k in data.keys() if "time" in k and test in k]
        back = [k for k in data.keys() if "time" not in k and test in k and "back" in k]
        right = [k for k in data.keys() if "time" not in k and test in k and "right" in k]
        front = [k for k in data.keys() if "time" not in k and test in k and "front" in k]
        left = [k for k in data.keys() if "time" not in k and test in k and "left" in k]
        return time_keyes, [back, right, front, left]

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
        #plt.show()

    def get_average_data(self, data, keys):
        filtered_data = [[k, self.average(data[k])] for k in keys if k in data and "0-2" not in k]
        return filtered_data

    def average(self, data):
        flatten = self.flatten(data)
        return sum(flatten)/len(flatten)

    def plot_average_both(self, data, keys1, keys2):
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
        #plt.show()

    def get_data_values(self,data, keys):
        filtered_data = {k:self.flatten(data[k]) for k in keys if k in data and "0-2" not in k and "Sade" not in k and "Jonas" not in k}
        return filtered_data

    def get_data_values_divided(self,data, keys):
        filtered_data = {k:self.flatten_divided(data[k]) for k in sorted(keys) if k in data and "Sade" not in k and "Jonas" not in k}
        return filtered_data

    def flatten_divided(self, values):
        return [p['probs'] for f in values for p in values[f]]

    def plot_violinplot(self, data, keys1, keys2):
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 5))

        # Test 1
        data1 = self.get_data_values(data, keys1)
        ax1.violinplot(data1.values())
        ax1.set_xticks(range(1,len(data1)+1))
        labels1 = ax1.set_xticklabels(data1.keys())
        for i, label in enumerate(labels1):
            label.set_y(label.get_position()[1] - (i % 2) * 0.075)
        ax1.set_title("Test 1")

        # Test 2
        data2 = self.get_data_values(data, keys2)
        ax2.violinplot(data2.values())
        ax2.set_xticks(range(1, len(data2) + 1))
        labels2 = ax2.set_xticklabels([s[:-6] for s in data2.keys()])
        for i, label in enumerate(labels2):
            label.set_y(label.get_position()[1] - (i % 2) * 0.075)
        ax2.set_title("Test 2")

        #plt.show()


    def plot_violinplot_orientation(self, data, keys, ax_num, titles):
        fig, axs = plt.subplots(ax_num[0], ax_num[1], figsize=(15, 10))
        i = 0
        j = 0
        for k in keys:
            data1 = self.get_data_values_divided(data, k )
            ax = axs[j,i%2]
            if len(data1) == 0:
                i += 1
                j = j + 1 if i % 2 == 0 else j
                continue
            ax.violinplot(data1.values())
            ax.set_xticks(range(1, len(data1) + 1))
            labels1 = ax.set_xticklabels([s[:-11] for s in data1.keys()])
            for l, label in enumerate(labels1):
                label.set_y(label.get_position()[1] - (l % 2) * 0.075)
            ax.set_title(titles[i])
            i += 1
            j = j + 1 if i % 2 == 0 else j

        #plt.show()

    def flatten_dict(self, values):
        return self.flatten([values[k] for k in values])

    def plot_violinplot_all(self, data, keys):
        plt.Figure(figsize=(6,6))
        data1 = self.flatten_dict(self.get_data_values_divided(data, keys[0]))
        data2 = self.flatten_dict(self.get_data_values_divided(data, keys[1]))
        plt.violinplot([data1, data2])
        plt.xticks(range(1, 3), ["Test 1", "Test 2"])
        #plt.show()


if __name__ == "__main__":
    AnalyseData().main()