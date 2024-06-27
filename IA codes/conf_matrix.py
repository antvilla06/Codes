import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Define sample labels
label_original = [2, 0, 58, 2, 4, 4, 1, 89, 24, 78, 14]
label_given = [25, 1, 17, 2, 4, 27, 1, 0, 1, 3, 3]

# Create confusion matrix
confusion_mat = confusion_matrix(label_original, label_given)

# Visualize confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()

# Classification report
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(label_original, label_given, target_names=targets))
