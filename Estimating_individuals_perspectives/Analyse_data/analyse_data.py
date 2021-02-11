import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu

class AnalyseData:
    def __init__(self):
        sns.set_theme(style="ticks", font_scale=1.2)
        self.directory = "../../Test_data/Test1_Test2/Results_hele_klip/results_test1_test2.json"
        self.directory_divided = "../../Test_data/Test1_Test2/Opdelt_30_Result/results_test1_test2.json"
        self.dir_results = ["face", "avg", "first", "30"]
        self.dir_input = ["Opdelt_face_Result", "Opdelt_avg_Result", "Opdelt_Result", "Opdelt_30_Result"]
        self.methods = ["1", "2", "3", "3a"]
        self.folder = ["face", "avg", "first", "30"]

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
        #self.plot_violinplot_orientation(data_divided, keys1, [2,2], ["","Right","Front","Left"], "All")
        #plt.savefig("../result/face/violin_test1.png")

        _, keys2 = self.get_orientation_keys_data(data_divided, "test2")
        #self.plot_violinplot_orientation(data_divided, keys2, [2,2],["Back","Right","Front","Left"], "All")
        #plt.savefig("../result/face/violin_test2.png")

        #self.plot_violinplot_orientation(data_divided, [keys1[1], keys2[1]] , [1,2],["Test 1","Test 2"], "Right")
        #plt.savefig("../result/first/violin_right.png")
        #self.plot_violinplot_orientation(data_divided, [keys1[2], keys2[2]], [1, 2], ["Test 1", "Test 2"], "Front")
        #plt.savefig("../result/first/violin_front.png")
        #self.plot_violinplot_orientation(data_divided, [keys1[3], keys2[3]], [1, 2], ["Test 1", "Test 2"], "Left")
        self.plot_violinplot_orientation(data_divided, [keys2[0]], [1, 1], ["Test 2"], "Back")
        plt.savefig("../result/30/violin_back.png")
        #plt.savefig("../result/first/violin_left.png")
        #plt.show()


        for i in range(len(self.dir_results)):
            dir = "../../Test_data/Test1_Test2/" + self.dir_input[i] + "/results_test1_test2.json"
            data_divided = self.get_data_dir(dir)

            _, keys1 = self.get_orientation_keys_data(data_divided, "test1")
            _, keys2 = self.get_orientation_keys_data(data_divided, "test2", False)
            self.plot_violinplot_all(data_divided, [self.flatten(keys1), self.flatten(keys2)], self.folder[i], self.methods[i])

            time, parts = self.get_time_keys_data(data_divided)
            self.plot_violinplot_time(data_divided, time, parts, self.methods[i])
            plt.show()

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

    def get_orientation_keys_data(self, data, test, with_back = True):
        time_keyes = [k for k in data.keys() if "time" in k and test in k]
        back = [k for k in data.keys() if "time" not in k and test in k and "back" in k] if with_back else []
        right = [k for k in data.keys() if "time" not in k and test in k and "right" in k]
        front = [k for k in data.keys() if "time" not in k and test in k and "front" in k]
        left = [k for k in data.keys() if "time" not in k and test in k and "left" in k]
        return time_keyes, [back, right, front, left]

    def get_time_keys_data(self, data):
        gaze = [k for k in data.keys() if "time" in k and "gaze360" in k]
        detecting = [k for k in data.keys() if "time" in k and "detectingAttendedTargets" in k]
        distribution = [k for k in data.keys() if "time" in k and "distribution" in k]
        time_keyes = [k for k in data.keys() if "-time" in k]
        return time_keyes, [gaze, detecting, distribution]

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
        # general info



    def plot_violinplot_orientation(self, data, keys, ax_num, titles, overall_title):
        size = (15, 10) if ax_num[0] > 1 else (17, 5)
        fig, axs = plt.subplots(ax_num[0], ax_num[1], figsize=size)
        i = 0
        j = 0
        for k in keys:
            data1 = self.get_data_values_divided(data, k )
            ax = axs[i] if ax_num[0] == 1 and ax_num[1] == 2 else axs if ax_num[0] == 1 else axs[j,i%2]
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
        fig.suptitle("Method 3a: "+overall_title)
        #plt.show()

    def flatten_dict(self, values):
        return self.flatten([values[k] for k in values])

    def plot_violinplot_all(self, data, keys, name, m_name, save=True):
        plt.show()
        plt.Figure(figsize=(6,6))
        data1 = self.flatten_dict(self.get_data_values_divided(data, keys[0]))
        data2 = self.flatten_dict(self.get_data_values_divided(data, keys[1]))
        plt.violinplot([data1, data2])
        plt.xticks(range(1, 3), ["Test 1", "Test 2"])
        plt.title("Method "+m_name)
        plt.ylabel("Probability Estimation")
        if save:
            plt.savefig("../result/" + name + "/violin_combined.png")

        print("----" + "Time for Method " + name)

        print(len(data1))
        print(np.mean(data1))
        print(np.median(data1))
        print(np.min(data1))
        print(np.max(data1))
        print("data2")
        print(len(data2))
        print(np.mean(data2))
        print(np.median(data2))
        print(np.min(data2))
        print(np.max(data2))
        print(np.sum(data1), np.sum(data2))
        data1 = np.array(data1)
        data2 = np.array(data2)
        print(len(data1[data1 > 0.5]), len(data2[data2 > 0.5]))
        stat, p = mannwhitneyu(data1, data2)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        alpha = 0.05
        if p > alpha:
            print('Same distribution (fail to reject H0)')
        else:
            print('Different distribution (reject H0)')

    def plot_violinplot_time(self, data, time, parts, name, save=True):
        plt.show()
        fig, axs = plt.subplots(1, 2, figsize=(8,5))
        data_time = [data[k] for k in time]
        axs[0].violinplot(data_time)
        axs[0].set_xticks(range(1, 2))
        axs[0].set_xticklabels(["Time"])
        axs[0].set_ylabel("Time in secunds")
        if save:
            plt.savefig("../result/time/m"+name+".png")

        #plt.show()
        plt.Figure(figsize=(6, 6))
        data1 = [data[k] for k in parts[0]]
        #data2 = [data[k] for k in parts[1]]
        data3 = [data[k] for k in parts[2]]
        axs[1].violinplot([data1, data3])
        axs[1].set_xticks(range(1, 3))
        axs[1].set_xticklabels(["Gaze360", "Distribution"])
        fig.suptitle("Time for Method " + name)
        if save:
            plt.savefig("../result/time/m"+name+"_divided.png")
        plt.show()

        print("----"+"Time for Method "+name)
        for d in [data_time, data1, data3]:
            print("new")
            print(len(d))
            print(np.mean(d))
            print(np.median(d))
            print(np.min(d))
            print(np.max(d))

if __name__ == "__main__":
    AnalyseData().main()