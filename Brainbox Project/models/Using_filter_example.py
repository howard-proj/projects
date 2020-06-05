import matplotlib.pyplot as plt
from scipy.io import wavfile
from gaussianFilter import filterData

# fileName = 'RLR.wav'
fileName = 'RLLRLRLRRRLR.wav'
sampleFreq, ampliData = wavfile.read(fileName)
data_filtered = filterData(ampliData)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

ax1.plot(ampliData) # Raw data
ax1.plot(data_filtered)
plt.show()
