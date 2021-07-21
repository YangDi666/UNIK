import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools

actions=['Cook.Cleandishes','Cook.Cleanup','Cook.Cut', 'Cook.Stir', 'Cook.Usestove', 
'Cutbread', 'Drink.Frombottle', 'Drink.Fromcan', 'Drink.Fromcup', 'Drink.Fromglass',
'Eat.Attable', 'Eat.Snack', 'Enter','Getup', 'Laydown', 'Leave', 
'Makecoffee.Pourgrains', 'Makecoffee.Pourwater', 'Maketea.Boilwater', 
'Maketea.Insertteabag', 'Pour.Frombottle', 
'Pour.Fromcan', 'Pour.Fromkettle', 'Readbook', 'Sitdown', 'Takepills', 
'Uselaptop', 'Usetelephone', 'Usetablet', 'Walk', 'WatchTV']


filename=sys.argv[1]
num_cls=int(sys.argv[2])

con_matrix=np.zeros((num_cls,num_cls))
accuracy_matrix=np.zeros((num_cls,num_cls))

with open(filename,'r') as result:
    results=result.readlines()
for i in results:
    predict=int(i.split(',')[0])
    gt=int(i.split(',')[1])
    con_matrix[gt, predict]+=1

sum_gt=con_matrix.sum(axis=1)
sum_t=sum_gt.sum(axis=0)
tp=0
acc_tot=0

for j, row in enumerate(con_matrix):
    accuracy_matrix[j]=row/sum_gt[j]
for a in range(num_cls):
    tp+=con_matrix[a,a]
    print('accuracy_'+actions[a]+' : ', accuracy_matrix[a,a])
    acc_tot+=accuracy_matrix[a,a]
print('\ntotal accuracy: ', tp/sum_t )
print('\naverage accuracy: ', acc_tot/(num_cls))

# plot confusion matrix
plt.imshow(accuracy_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(actions))
plt.xticks(tick_marks, actions, rotation=90)
plt.yticks(tick_marks, actions)
plt.ylim(len(accuracy_matrix) - 0.5, -0.5)
thresh = accuracy_matrix.max() / 2.
for i, j in itertools.product(range(accuracy_matrix.shape[0]), range(accuracy_matrix.shape[1])):
    if i==j:
        plt.text(j, i, '{:.2f}'.format(accuracy_matrix[i, j]), horizontalalignment="center",
        color="white" if accuracy_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.18)
plt.ylabel('True label')
plt.xlabel('Predicted label')
  
plt.show()
plt.savefig('confusion_matrix.png')
print('Figure saved as confusion_matrix.png')
