import matplotlib.pyplot as plt
import seaborn as sns
with open('acuracy.txt', "rb") as f:
    x1=f.read()

time = range(500)

sns.set(style="darkgrid", font_scale=1.5)
sns.tsplot(time=time, data=x1, color="r", condition="behavior_cloning")
#sns.tsplot(time=time, data=x2, color="b", condition="dagger")

plt.ylabel("Reward")
plt.xlabel("Iteration Number")
plt.title("Imitation Learning")

plt.show()