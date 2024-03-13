import matplotlib.pyplot as plt

import json

with open("C:\\Users\\l-u-u\\PycharmProjects\\gasten\\src\\optimization\\stats.json") as json_file:
    json_data = json.load(json_file)

plt.plot([1,2], json_data['eval']['fid'])
plt.plot([1,2],[
      316.94786412757765,
      393.54596513535523
    ])
plt.plot([1,2,3,4,5], [
      150.94786412757765,
      80.54596513535523,
    60,
    40,
    23
    ])
plt.plot([1,2],[
      320.7896332357163,
      322.6901107373732
    ])
plt.show()