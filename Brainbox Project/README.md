Project associated with REM Sleep at University of Sydney

Sleep studies have shown that poor sleeping patterns correlate to poorer future mental health in individuals. Therefore today, it's more relevant than ever to study changing sleep stages which constitute these sleep patterns as well as methods used to detect these sleep stages, including the electrooculogram (EOG) often used to detect REM sleep. However, although such techniques conducted in controlled scientific environments can accurately track a subject through the stages of sleep, in uncontrolled environments it's often difficult to attain the same level of preciseness, mostly due to variation in calibration and environmental noise. This limits the people who can study these phenomena, thus limiting scientific development.

Therefore, through software that can visualise varying input parameters on output metrics, thereby testing the model's robustness to noise, users are provided with an invaluable tool to attain near optimal parameters given some signal with its implied range of environmental variation. To obtain the near optimal parameters, a configuration region of interest should be specified by the user to then randomly iterate within this space, yielding performance of the configurated model at each step.

The software was implemented towards signal isolation of eye movement events from an EOG. However, the concept may be generalised to include any time sequential event isolation problems, albeit requiring it's own ground truth generation algorithm. In this case, due to resource constraints, the concept was demonstrated using an SVM classifier alongside four input hyperparameters: classification window, sample proportion, cleaner window and cleaner threshold.

Through a 4D visualization, optimization of REM Sleep can be achieved. Simply record a waveform data, or use any of the data in this project for analysis. Randomize your data to visualize the best optimization!

Choose the waveform data in the assets directory. Once saved, the data will be saved in saved directory. The naming convention follows the waveform data with .remdata as its extension. Upon loading the data, ensure the correct classifier is chosen from the original savepoint otherwise something cool will happen!




Note: Mind the number of randmozation may take some time depending on the number of iterations
